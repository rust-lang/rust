use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;

#[test]
fn is_integer() {
    let t = Double::from_f64(-0.0);
    assert!(t.is_integer());
    let t = Double::from_f64(3.14159);
    assert!(!t.is_integer());
    let t = Double::NAN;
    assert!(!t.is_integer());
    let t = Double::INFINITY;
    assert!(!t.is_integer());
    let t = -Double::INFINITY;
    assert!(!t.is_integer());
    let t = Double::largest();
    assert!(t.is_integer());
}

#[test]
fn is_negative() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_negative());
    let t = "-0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_negative());

    assert!(!Single::INFINITY.is_negative());
    assert!((-Single::INFINITY).is_negative());

    assert!(!Single::ZERO.is_negative());
    assert!((-Single::ZERO).is_negative());

    assert!(!Single::NAN.is_negative());
    assert!((-Single::NAN).is_negative());

    assert!(!Single::snan(None).is_negative());
    assert!((-Single::snan(None)).is_negative());
}

#[test]
fn is_normal() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_normal());

    assert!(!Single::INFINITY.is_normal());
    assert!(!Single::ZERO.is_normal());
    assert!(!Single::NAN.is_normal());
    assert!(!Single::snan(None).is_normal());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_normal());
}

#[test]
fn is_finite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_finite());
    assert!(!Single::INFINITY.is_finite());
    assert!(Single::ZERO.is_finite());
    assert!(!Single::NAN.is_finite());
    assert!(!Single::snan(None).is_finite());
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite());
}

#[test]
fn is_finite_non_zero() {
    // Test positive/negative normal value.
    assert!("0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());

    // Test positive/negative denormal value.
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());

    // Test +/- Infinity.
    assert!(!Single::INFINITY.is_finite_non_zero());
    assert!(!(-Single::INFINITY).is_finite_non_zero());

    // Test +/- Zero.
    assert!(!Single::ZERO.is_finite_non_zero());
    assert!(!(-Single::ZERO).is_finite_non_zero());

    // Test +/- qNaN. +/- don't mean anything with qNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::NAN.is_finite_non_zero());
    assert!(!(-Single::NAN).is_finite_non_zero());

    // Test +/- sNaN. +/- don't mean anything with sNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::snan(None).is_finite_non_zero());
    assert!(!(-Single::snan(None)).is_finite_non_zero());
}

#[test]
fn is_infinite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_infinite());
    assert!(Single::INFINITY.is_infinite());
    assert!(!Single::ZERO.is_infinite());
    assert!(!Single::NAN.is_infinite());
    assert!(!Single::snan(None).is_infinite());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_infinite());
}

#[test]
fn is_nan() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_nan());
    assert!(!Single::INFINITY.is_nan());
    assert!(!Single::ZERO.is_nan());
    assert!(Single::NAN.is_nan());
    assert!(Single::snan(None).is_nan());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_nan());
}

#[test]
fn is_signaling() {
    // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads.
    let payload = 4;
    assert!(!Single::qnan(None).is_signaling());
    assert!(!(-Single::qnan(None)).is_signaling());
    assert!(!Single::qnan(Some(payload)).is_signaling());
    assert!(!(-Single::qnan(Some(payload))).is_signaling());
    assert!(Single::snan(None).is_signaling());
    assert!((-Single::snan(None)).is_signaling());
    assert!(Single::snan(Some(payload)).is_signaling());
    assert!((-Single::snan(Some(payload))).is_signaling());
}

#[test]
fn nan() {
    fn nanbits<T: Float>(signaling: bool, negative: bool, fill: u128) -> u128 {
        let x = if signaling { T::snan(Some(fill)) } else { T::qnan(Some(fill)) };
        if negative { (-x).to_bits() } else { x.to_bits() }
    }

    assert_eq!(0x7fc00000, nanbits::<Single>(false, false, 0));
    assert_eq!(0xffc00000, nanbits::<Single>(false, true, 0));
    assert_eq!(0x7fc0ae72, nanbits::<Single>(false, false, 0xae72));
    assert_eq!(0x7fffae72, nanbits::<Single>(false, false, 0xffffae72));
    assert_eq!(0x7fa00000, nanbits::<Single>(true, false, 0));
    assert_eq!(0xffa00000, nanbits::<Single>(true, true, 0));
    assert_eq!(0x7f80ae72, nanbits::<Single>(true, false, 0xae72));
    assert_eq!(0x7fbfae72, nanbits::<Single>(true, false, 0xffffae72));

    assert_eq!(0x7ff8000000000000, nanbits::<Double>(false, false, 0));
    assert_eq!(0xfff8000000000000, nanbits::<Double>(false, true, 0));
    assert_eq!(0x7ff800000000ae72, nanbits::<Double>(false, false, 0xae72));
    assert_eq!(0x7fffffffffffae72, nanbits::<Double>(false, false, 0xffffffffffffae72));
    assert_eq!(0x7ff4000000000000, nanbits::<Double>(true, false, 0));
    assert_eq!(0xfff4000000000000, nanbits::<Double>(true, true, 0));
    assert_eq!(0x7ff000000000ae72, nanbits::<Double>(true, false, 0xae72));
    assert_eq!(0x7ff7ffffffffae72, nanbits::<Double>(true, false, 0xffffffffffffae72));
}

#[test]
fn neg() {
    let one = "1.0".parse::<Single>().unwrap();
    let neg_one = "-1.0".parse::<Single>().unwrap();
    let zero = Single::ZERO;
    let neg_zero = -Single::ZERO;
    let inf = Single::INFINITY;
    let neg_inf = -Single::INFINITY;
    let qnan = Single::NAN;
    let neg_qnan = -Single::NAN;

    assert!(neg_one.bitwise_eq(-one));
    assert!(one.bitwise_eq(-neg_one));
    assert!(neg_zero.bitwise_eq(-zero));
    assert!(zero.bitwise_eq(-neg_zero));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_qnan.bitwise_eq(-qnan));
    assert!(qnan.bitwise_eq(-neg_qnan));
}
