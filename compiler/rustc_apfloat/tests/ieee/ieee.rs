use rustc_apfloat::ieee::{Double, Half, Quad, Single, X87DoubleExtended};
use rustc_apfloat::Float;

trait SingleExt {
    fn from_f32(input: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl SingleExt for Single {
    fn from_f32(input: f32) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f32(self) -> f32 {
        f32::from_bits(self.to_bits() as u32)
    }
}

trait DoubleExt {
    fn from_f64(input: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl DoubleExt for Double {
    fn from_f64(input: f64) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f64(self) -> f64 {
        f64::from_bits(self.to_bits() as u64)
    }
}

#[test]
fn decimal_strings_without_null_terminators() {
    // Make sure that we can parse strings without null terminators.
    // rdar://14323230.
    let val = "0.00"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.01"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.09"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.095"[..4].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.09);
    let val = "0.00e+3"[..7].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.00);
    let val = "0e+3"[..4].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.00);
}

#[test]
fn exact_inverse() {
    // Trivial operation.
    assert!(Double::from_f64(2.0).get_exact_inverse().unwrap().bitwise_eq(Double::from_f64(0.5)));
    assert!(Single::from_f32(2.0).get_exact_inverse().unwrap().bitwise_eq(Single::from_f32(0.5)));
    assert!(
        "2.0"
            .parse::<Quad>()
            .unwrap()
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq("0.5".parse::<Quad>().unwrap())
    );
    assert!(
        "2.0"
            .parse::<X87DoubleExtended>()
            .unwrap()
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq("0.5".parse::<X87DoubleExtended>().unwrap())
    );

    // FLT_MIN
    assert!(
        Single::from_f32(1.17549435e-38)
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq(Single::from_f32(8.5070592e+37))
    );

    // Large float, inverse is a denormal.
    assert!(Single::from_f32(1.7014118e38).get_exact_inverse().is_none());
    // Zero
    assert!(Double::from_f64(0.0).get_exact_inverse().is_none());
    // Denormalized float
    assert!(Single::from_f32(1.40129846e-45).get_exact_inverse().is_none());
}

#[test]
fn largest() {
    assert_eq!(3.402823466e+38, Single::largest().to_f32());
    assert_eq!(1.7976931348623158e+308, Double::largest().to_f64());
}

#[test]
fn smallest() {
    let test = Single::SMALLEST;
    let expected = "0x0.000002p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Single::SMALLEST;
    let expected = "-0x0.000002p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = Quad::SMALLEST;
    let expected = "0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Quad::SMALLEST;
    let expected = "-0x0.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn smallest_normalized() {
    let test = Single::smallest_normalized();
    let expected = "0x1p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Single::smallest_normalized();
    let expected = "-0x1p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = Quad::smallest_normalized();
    let expected = "0x1p-16382".parse::<Quad>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Quad::smallest_normalized();
    let expected = "-0x1p-16382".parse::<Quad>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn zero() {
    assert_eq!(0.0, Single::from_f32(0.0).to_f32());
    assert_eq!(-0.0, Single::from_f32(-0.0).to_f32());
    assert!(Single::from_f32(-0.0).is_negative());

    assert_eq!(0.0, Double::from_f64(0.0).to_f64());
    assert_eq!(-0.0, Double::from_f64(-0.0).to_f64());
    assert!(Double::from_f64(-0.0).is_negative());

    fn test<T: Float>(sign: bool, bits: u128) {
        let test = if sign { -T::ZERO } else { T::ZERO };
        let pattern = if sign { "-0x0p+0" } else { "0x0p+0" };
        let expected = pattern.parse::<T>().unwrap();
        assert!(test.is_zero());
        assert_eq!(sign, test.is_negative());
        assert!(test.bitwise_eq(expected));
        assert_eq!(bits, test.to_bits());
    }
    test::<Half>(false, 0);
    test::<Half>(true, 0x8000);
    test::<Single>(false, 0);
    test::<Single>(true, 0x80000000);
    test::<Double>(false, 0);
    test::<Double>(true, 0x8000000000000000);
    test::<Quad>(false, 0);
    test::<Quad>(true, 0x8000000000000000_0000000000000000);
    test::<X87DoubleExtended>(false, 0);
    test::<X87DoubleExtended>(true, 0x8000_0000000000000000);
}

#[test]
fn copy_sign() {
    assert!(
        Double::from_f64(-42.0)
            .bitwise_eq(Double::from_f64(42.0).copy_sign(Double::from_f64(-1.0),),)
    );
    assert!(
        Double::from_f64(42.0)
            .bitwise_eq(Double::from_f64(-42.0).copy_sign(Double::from_f64(1.0),),)
    );
    assert!(
        Double::from_f64(-42.0)
            .bitwise_eq(Double::from_f64(-42.0).copy_sign(Double::from_f64(-1.0),),)
    );
    assert!(
        Double::from_f64(42.0)
            .bitwise_eq(Double::from_f64(42.0).copy_sign(Double::from_f64(1.0),),)
    );
}

#[test]
fn operator_overloads() {
    // This is mostly testing that these operator overloads compile.
    let one = "0x1p+0".parse::<Single>().unwrap();
    let two = "0x2p+0".parse::<Single>().unwrap();
    assert!(two.bitwise_eq((one + one).value));
    assert!(one.bitwise_eq((two - one).value));
    assert!(two.bitwise_eq((one * two).value));
    assert!(one.bitwise_eq((two / two).value));
}
