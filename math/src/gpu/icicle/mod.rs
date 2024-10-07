use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::{
                bls12_377::curve::BLS12377Curve,
                bls12_381::{
                    curve::BLS12381Curve, default_types::{FrField, FrElement},
                    field_extension::BLS12381PrimeField, 
                },
                bn_254::{curve::BN254Curve, field_extension::BN254PrimeField},
                grumpkin::{curve::{GrumpkinCurve, GrumpkinPrimeField}},
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::IsEllipticCurve,
    },
    errors::ByteConversionError,
    fft::errors::FFTError,
    field::{
        element::FieldElement,
        fields::fft_friendly::{stark_252_prime_field::Stark252PrimeField, babybear::Babybear31PrimeField},
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    msm::naive::MSMError,
    polynomial::Polynomial,
    traits::ByteConversion,
};
use icicle_bls12_377::curve::CurveCfg as IcicleBLS12377Curve;
use icicle_bls12_381::curve::{
    CurveCfg as IcicleBLS12381Curve, ScalarCfg as IcicleBLS12381ScalarCfg,
};
use icicle_bn254::curve::{CurveCfg as IcicleBN254Curve, ScalarCfg as IcicleBN254ScalarCfg};
use icicle_grumpkin::curve::{CurveCfg as IcicleGrumpkinCurve, ScalarCfg as IcicleGrumpkinScalarCfg};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    ntt::{ntt_inplace, NTTConfig, NTTDomain, NTTDir,},
    traits::FieldImpl,
};
use icicle_runtime::{memory::{HostOrDeviceSlice, DeviceVec, HostSlice}, stream::IcicleStream};

use std::fmt::Debug;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl IcicleMSM for ShortWeierstrassProjectivePoint<BLS12381Curve> {
    type Curve = IcicleBLS12381Curve;

    fn curve_name() -> &'static str {
        "BLS12381"
    }

    fn to_icicle_affine(&self) -> Affine<Self::Curve> {
        let s = self.to_affine();
        Affine::<Self::Curve> {
            x: Self::to_icicle_field(s.x()),
            y: Self::to_icicle_field(s.y()),
        }
    }

    fn from_icicle_projective(
        icicle: &Projective<Self::Curve>,
    ) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            Self::from_icicle_field(&icicle.x).unwrap(),
            Self::from_icicle_field(&icicle.y).unwrap(),
            Self::from_icicle_field(&icicle.z).unwrap(),
        ]))
    }
}

impl IcicleMSM for ShortWeierstrassProjectivePoint<BLS12377Curve> {
    type Curve = IcicleBLS12377Curve;
    fn curve_name() -> &'static str {
        "BLS12377"
    }

    fn to_icicle_affine(&self) -> Affine<Self::Curve> {
        let s = self.to_affine();
        Affine::<Self::Curve> {
            x: Self::to_icicle_field(s.x()),
            y: Self::to_icicle_field(s.y()),
        }
    }

    fn from_icicle_projective(
        icicle: &Projective<Self::Curve>,
    ) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            Self::from_icicle_field(&icicle.x).unwrap(),
            Self::from_icicle_field(&icicle.y).unwrap(),
            Self::from_icicle_field(&icicle.z).unwrap(),
        ]))
    }
}

impl IcicleMSM for ShortWeierstrassProjectivePoint<BN254Curve> {
    type Curve = IcicleBN254Curve;
    fn curve_name() -> &'static str {
        "BN254"
    }

    fn to_icicle_affine(&self) -> Affine<Self::Curve> {
        let s = self.to_affine();
        Affine::<Self::Curve> {
            x: Self::to_icicle_field(s.x()),
            y: Self::to_icicle_field(s.y()),
        }
    }

    fn from_icicle_projective(
        icicle: &Projective<Self::Curve>,
    ) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            Self::from_icicle_field(&icicle.x).unwrap(),
            Self::from_icicle_field(&icicle.y).unwrap(),
            Self::from_icicle_field(&icicle.z).unwrap(),
        ]))
    }
}

pub trait IcicleMSM: IsGroup {
    type Curve: Curve + MSM<Self::Curve>;
    //type FE: ByteConversion;
    /// Used for searching this field's implementation in other languages, e.g in MSL
    /// for executing parallel operations with the Metal API.
    fn curve_name() -> &'static str {
        ""
    }

    fn to_icicle_affine(&self) -> Affine<Self::Curve>;

    fn from_icicle_projective(
        icicle: &Projective<Self::Curve>,
    ) -> Result<Self, ByteConversionError>;

    fn to_icicle_field<FE: ByteConversion>(element: &FE) -> <Self::Curve as Curve>::BaseField {
        <Self::Curve as Curve>::BaseField::from_bytes_le(&element.to_bytes_le())
    }

    fn to_icicle_scalar<FE: ByteConversion>(
        element: &FE,
    ) -> <Self::Curve as Curve>::ScalarField {
        <Self::Curve as Curve>::ScalarField::from_bytes_le(&element.to_bytes_le())
    }

    fn from_icicle_scalar<FE: ByteConversion>(
        element: &<Self::Curve as Curve>::ScalarField,
    ) -> Result<FE, ByteConversionError> {
        FE::from_bytes_le(&element.to_bytes_le())
    }

    fn from_icicle_field<FE: ByteConversion>(
        icicle: &<Self::Curve as Curve>::BaseField,
    ) -> Result<FE, ByteConversionError> {
        FE::from_bytes_le(&icicle.to_bytes_le())
    }
}

//TODO: consider removing these traits functions for explicitness or extrapolating into another SubTrait
//TODO: Good news is the conversion and NTT interface is much simpler. 
//  - Bad news is we still will need to map the extension fields.
pub trait IcicleFFT: IsField
where
    FieldElement<Self>: ByteConversion,
{
    type ScalarField: FieldImpl + NTTDomain<Self::ScalarField>;

    fn to_icicle_scalar(element: &FieldElement<Self>) -> Self::ScalarField {
        Self::ScalarField::from_bytes_le(&element.to_bytes_le())
    }

    fn from_icicle_scalar(
        icicle: &Self::ScalarField,
    ) -> Result<FieldElement<Self>, ByteConversionError> {
        FieldElement::<Self>::from_bytes_le(&icicle.to_bytes_le())
    }
}

/*
impl IcicleFFT for BLS12381PrimeField {
    type ScalarField = <IcicleBLS12381Curve as Curve>::ScalarField;
}

// For BLS12381 -> TODO Later
/*
impl IcicleFFT for FrField {
    type ScalarField = <IcicleBLS12381Curve as Curve>::ScalarField;
}
*/

impl IcicleFFT for BN254PrimeField {
    type ScalarField = <IcicleBN254Curve as Curve>::ScalarField;
}

// DUMMY IMPLEMENTATION OF STARK252 -> Fails when Icicle feature flag enabled
impl IcicleFFT for Stark252PrimeField {
    type ScalarField = <IcicleBLS12381Curve as Curve>::ScalarField;
}
*/


//Note bases should be converted in parallel outside of this function.
//TODO: add optional bit size
pub fn icicle_msm<F: IsField, G: IcicleMSM>(
    bases: &[Affine<<G as IcicleMSM>::Curve>],
    scalars: &[FieldElement<F>],
) -> Result<G, MSMError>
where
    FieldElement<F>: ByteConversion,
{
    let mut stream = IcicleStream::create().unwrap();
    //let mut scalars_slice = DeviceVec::<<<G as IcicleMSM>::Curve as Curve>::ScalarField>::device_malloc(scalars.len()).unwrap();
    /*
    let scalars: Vec<<<G as IcicleMSM>::Curve as Curve>::ScalarField> = 
        scalars.iter()
            .map(|scalar| G::to_icicle_scalar(scalar))
            .collect::<Vec<_>>();
    */
    //We directly transmute the scalars this significantly speeds up the operations. See this test made by Icicle.
    let scalars = unsafe { &*(&scalars[..] as *const _ as *const [<<G as IcicleMSM>::Curve as Curve>::ScalarField]) };
    //scalars_slice.copy_from_host(HostSlice::from_slice(&scalars[..])).unwrap();
    //let mut bases_slice = DeviceVec::<Affine<<G as IcicleMSM>::Curve>>::device_malloc(bases.len()).unwrap();
    //bases_slice.copy_from_host(HostSlice::from_slice(&bases[..])).unwrap();
    /*
    let points = HostOrDeviceSlice::Host(
        points
            .iter()
            .map(|point| G::to_icicle_affine(point))
            .collect::<Vec<_>>(),
    );
    */

    let mut msm_result = DeviceVec::<Projective<<G as IcicleMSM>::Curve>>::device_malloc(1).unwrap();
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = *stream;
    cfg.is_async = false;
    cfg.are_scalars_montgomery_form = false;
    cfg.are_bases_montgomery_form = false;

    let mut msm_host_result = vec![Projective::<<G as IcicleMSM>::Curve>::zero()];
    msm(HostSlice::from_slice(&scalars[..]), HostSlice::from_slice(&bases[..]), &cfg, HostSlice::from_mut_slice(&mut msm_host_result)).unwrap();

    msm_result.copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..])).unwrap();
    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    G::from_icicle_projective(&msm_host_result[0]).map_err(|e| MSMError::ConversionError(e))
}

/*
pub fn evaluate_fft_icicle<F, E>(
    coeffs: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E> + IcicleFFT,
    FieldElement<E>: ByteConversion,
    E: IsField,
{

    let ntt_cfg: NTTConfig<<F as IcicleFFT>::ScalarField> = NTTConfig::default();
    ntt_inplace(HostSlice::from_mut_slice(&mut scalars[..]), NTTDir::kForward, &ntt_cfg).unwrap();

    /*
    let size = coeffs.len();
    let mut cfg = NTTConfig::default();
    let order = coeffs.len() as u64;
    let dir = NTTDir::kForward;
    let scalars = HostOrDeviceSlice::Host(
        coeffs
            .iter()
            .map(|scalar| E::to_icicle_scalar(&scalar))
            .collect::<Vec<_>>(),
    );
    let mut ntt_results = HostOrDeviceSlice::cuda_malloc(size).unwrap();
    let stream = IcicleStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    let root_of_unity = E::to_icicle_scalar(
        &(F::get_primitive_root_of_unity(order).unwrap() * FieldElement::<E>::one()),
    );
    <E as IcicleFFT>::Config::initialize_domain(root_of_unity, &cfg.ctx).unwrap();
    <E as IcicleFFT>::Config::ntt_unchecked(&scalars, dir, &cfg, &mut ntt_results).unwrap();
    stream.synchronize().unwrap();
    let mut ntt_host_results = vec![E::ScalarField::zero(); size];
    ntt_results.copy_to_host(&mut ntt_host_results[..]).unwrap();
    stream.destroy().unwrap();
    let res = ntt_host_results
        .as_slice()
        .iter()
        .map(|scalar| E::from_icicle_scalar(&scalar).unwrap())
        .collect::<Vec<_>>();
    Ok(res)
    */
}

pub fn interpolate_fft_icicle<F, E>(
    coeffs: &[FieldElement<E>],
) -> Result<Polynomial<FieldElement<E>>, FFTError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    FieldElement<E>: ByteConversion,
    E: IsField + IcicleFFT,
{
    /*
    let size = coeffs.len();
    let mut cfg = NTTConfig::default();
    let order = coeffs.len() as u64;
    let dir = NTTDir::kInverse;
    let scalars = HostOrDeviceSlice::Host(
        coeffs
            .iter()
            .map(|scalar| E::to_icicle_scalar(scalar))
            .collect::<Vec<_>>(),
    );
    let mut ntt_results = HostOrDeviceSlice::cuda_malloc(size).unwrap();
    let stream = IcicleStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    let root_of_unity = E::to_icicle_scalar(
        &(F::get_primitive_root_of_unity(order).unwrap() * FieldElement::<E>::one()),
    );
    <E as IcicleFFT>::Config::initialize_domain(root_of_unity, &cfg.ctx).unwrap();
    <E as IcicleFFT>::Config::ntt_unchecked(&scalars, dir, &cfg, &mut ntt_results).unwrap();
    stream.synchronize().unwrap();
    let mut ntt_host_results = vec![E::ScalarField::zero(); size];
    ntt_results.copy_to_host(&mut ntt_host_results[..]).unwrap();
    stream.destroy().unwrap();
    let res = ntt_host_results
        .as_slice()
        .iter()
        .map(|scalar| E::from_icicle_scalar(&scalar).unwrap())
        .collect::<Vec<_>>();
    */
    Ok(Polynomial::new(&res))
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement,
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        msm::pippenger::msm,
    };

    pub type Fr = FrElement;
    pub type G = ShortWeierstrassProjectivePoint<BLS12381Curve>;

    impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
        fn from_icicle_affine(
            icicle: &Affine<IcicleBLS12381Curve>,
        ) -> Result<ShortWeierstrassProjectivePoint<BLS12381Curve>, ByteConversionError> {
            Ok(Self::new([
                FieldElement::<BLS12381PrimeField>::from_bytes_le(&icicle.x.to_bytes_le()).unwrap(),
                FieldElement::<BLS12381PrimeField>::from_bytes_le(&icicle.y.to_bytes_le()).unwrap(),
                FieldElement::one(),
            ]))
        }
    }

    impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
        fn to_icicle_projective(
            &self,
        ) -> Projective<IcicleBLS12381Curve> {
            Projective {
                x: Self::to_icicle_field(self.x()),
                y: Self::to_icicle_field(self.y()),
                z: Self::to_icicle_field(self.z())
        }
        }
    }


    fn point_times_5() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = BLS12381FieldElement::from_hex_unchecked(
            "32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c",
        );
        let y = BLS12381FieldElement::from_hex_unchecked(
            "11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88",
        );
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn to_from_icicle_affine() {
        // convert value of 5 to icicle and back again and that icicle 5 matches
        let point = point_times_5();
        let icicle_point = point.to_icicle_affine();
        let res =
            G::from_icicle_affine(&icicle_point)
                .unwrap();
        assert_eq!(point, res)
    }

    #[test]
    fn to_from_icicle_projective() {
        // convert value of 5 to icicle and back again and that icicle 5 matches
        let point = point_times_5();
        //let icicle_point = point.to_icicle_affine();
        let icicle_projective =
            G::to_icicle_projective(&point);
        let res =
            G::from_icicle_projective(&icicle_projective)
                .unwrap();
        assert_eq!(point, res)
    }

    #[test]
    fn to_from_icicle_scalar() {
        // convert value of 5 to icicle and back again and that icicle 5 matches
        let scalar = Fr::from(8);
        let icicle_scalar = G::to_icicle_scalar(&scalar);
        let res = G::from_icicle_scalar(&icicle_scalar).unwrap();
        assert_eq!(scalar, res)
    }

    #[test]
    fn to_from_icicle_generator() {
        // Convert generator and see that it matches
        let point = BLS12381Curve::generator();
        let icicle_point = point.to_icicle_affine();
        let res =
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_icicle_affine(&icicle_point)
                .unwrap();
        assert_eq!(point, res)
    }

    #[test]
    fn icicle_g1_msm() {
        const LEN: usize = 1;
        let scalar: Fr = Fr::from(1);
        let lambda_scalars = vec![scalar; LEN];
        let lambda_points = (0..LEN).map(|_| point_times_5()).collect::<Vec<_>>();
        let expected = msm(&lambda_scalars, &lambda_points).unwrap();
        println!("Lambda Expected Affine: {:?}", expected.to_affine());
        let icicle_points = lambda_points.par_iter().map(|base| base.to_icicle_affine()).collect::<Vec<_>>();
        let res: G = icicle_msm(&icicle_points, &lambda_scalars).unwrap();
        assert_eq!(res, expected);
    }
}
