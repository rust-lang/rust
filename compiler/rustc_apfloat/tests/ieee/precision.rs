use rustc_apfloat::ieee::{Double, Quad, Single, X87DoubleExtended};
use rustc_apfloat::Round;

#[test]
fn fma() {
    {
        let mut f1 = Single::from_f32(14.5);
        let f2 = Single::from_f32(-14.5);
        let f3 = Single::from_f32(225.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(14.75, f1.to_f32());
    }

    {
        let val2 = Single::from_f32(2.0);
        let mut f1 = Single::from_f32(1.17549435e-38);
        let mut f2 = Single::from_f32(1.17549435e-38);
        f1 /= val2;
        f2 /= val2;
        let f3 = Single::from_f32(12.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(12.0, f1.to_f32());
    }

    // Test for correct zero sign when answer is exactly zero.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(!f1.is_negative() && f1.is_zero());
    }

    // Test for correct zero sign when answer is exactly zero and rounding towards
    // negative.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add_r(f2, f3, Round::TowardNegative).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test for correct (in this case -ve) sign when adding like signed zeros.
    // Test fma(0.0, -0.0, -0.0) -> -ve 0.
    {
        let mut f1 = Double::from_f64(0.0);
        let f2 = Double::from_f64(-0.0);
        let f3 = Double::from_f64(-0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test -ve sign preservation when small negative results underflow.
    {
        let mut f1 = "-0x1p-1074".parse::<Double>().unwrap();
        let f2 = "+0x1p-1074".parse::<Double>().unwrap();
        let f3 = Double::from_f64(0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test x87 extended precision case from http://llvm.org/PR20728.
    {
        let mut m1 = X87DoubleExtended::from_u128(1).value;
        let m2 = X87DoubleExtended::from_u128(1).value;
        let a = X87DoubleExtended::from_u128(3).value;

        let mut loses_info = false;
        m1 = m1.mul_add(m2, a).value;
        let r: Single = m1.convert(&mut loses_info).value;
        assert!(!loses_info);
        assert_eq!(4.0, r.to_f32());
    }
}

#[test]
fn min_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(1.0, f1.min(f2).to_f64());
    assert_eq!(1.0, f2.min(f1).to_f64());
    assert_eq!(1.0, f1.min(nan).to_f64());
    assert_eq!(1.0, nan.min(f1).to_f64());
}

#[test]
fn max_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(2.0, f1.max(f2).to_f64());
    assert_eq!(2.0, f2.max(f1).to_f64());
    assert_eq!(1.0, f1.max(nan).to_f64());
    assert_eq!(1.0, nan.max(f1).to_f64());
}

#[test]
fn denormal() {
    // Test single precision
    {
        assert!(!Single::from_f32(0.0).is_denormal());

        let mut t = "1.17549435082228750797e-38".parse::<Single>().unwrap();
        assert!(!t.is_denormal());

        let val2 = Single::from_f32(2.0e0);
        t /= val2;
        assert!(t.is_denormal());
    }

    // Test double precision
    {
        assert!(!Double::from_f64(0.0).is_denormal());

        let mut t = "2.22507385850720138309e-308".parse::<Double>().unwrap();
        assert!(!t.is_denormal());

        let val2 = Double::from_f64(2.0e0);
        t /= val2;
        assert!(t.is_denormal());
    }

    // Test Intel double-ext
    {
        assert!(!X87DoubleExtended::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626e-4932".parse::<X87DoubleExtended>().unwrap();
        assert!(!t.is_denormal());

        t /= X87DoubleExtended::from_u128(2).value;
        assert!(t.is_denormal());
    }

    // Test quadruple precision
    {
        assert!(!Quad::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626267781732175260e-4932".parse::<Quad>().unwrap();
        assert!(!t.is_denormal());

        t /= Quad::from_u128(2).value;
        assert!(t.is_denormal());
    }
}

#[test]
fn round_to_integral() {
    let t = Double::from_f64(-0.5);
    assert_eq!(-0.0, t.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(-1.0, t.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(-0.0, t.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(-0.0, t.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let s = Double::from_f64(3.14);
    assert_eq!(3.0, s.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(3.0, s.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(4.0, s.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(3.0, s.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let r = Double::largest();
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardNegative).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::TowardPositive).value.to_f64());
    assert_eq!(r.to_f64(), r.round_to_integral(Round::NearestTiesToEven).value.to_f64());

    let p = Double::ZERO.round_to_integral(Round::TowardZero).value;
    assert_eq!(0.0, p.to_f64());
    let p = (-Double::ZERO).round_to_integral(Round::TowardZero).value;
    assert_eq!(-0.0, p.to_f64());
    let p = Double::NAN.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_nan());
    let p = Double::INFINITY.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_infinite() && p.to_f64() > 0.0);
    let p = (-Double::INFINITY).round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_infinite() && p.to_f64() < 0.0);
}
