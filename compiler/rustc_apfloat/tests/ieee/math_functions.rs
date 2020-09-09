use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc_apfloat::{ExpInt, IEK_INF, IEK_NAN, IEK_ZERO};

#[test]
fn abs() {
    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let p_snan = Single::snan(None);
    let m_snan = -Single::snan(None);
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    assert!(p_inf.bitwise_eq(p_inf.abs()));
    assert!(p_inf.bitwise_eq(m_inf.abs()));
    assert!(p_zero.bitwise_eq(p_zero.abs()));
    assert!(p_zero.bitwise_eq(m_zero.abs()));
    assert!(p_qnan.bitwise_eq(p_qnan.abs()));
    assert!(p_qnan.bitwise_eq(m_qnan.abs()));
    assert!(p_snan.bitwise_eq(p_snan.abs()));
    assert!(p_snan.bitwise_eq(m_snan.abs()));
    assert!(p_normal_value.bitwise_eq(p_normal_value.abs()));
    assert!(p_normal_value.bitwise_eq(m_normal_value.abs()));
    assert!(p_largest_value.bitwise_eq(p_largest_value.abs()));
    assert!(p_largest_value.bitwise_eq(m_largest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(p_smallest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(m_smallest_value.abs()));
    assert!(p_smallest_normalized.bitwise_eq(p_smallest_normalized.abs(),));
    assert!(p_smallest_normalized.bitwise_eq(m_smallest_normalized.abs(),));
}

#[test]
fn ilogb() {
    assert_eq!(-1074, Double::SMALLEST.ilogb());
    assert_eq!(-1074, (-Double::SMALLEST).ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1024".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-51, "0x1p-51".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.c60f120d9f87cp-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(-2, "0x0.ffffp-1".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.fffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(1023, Double::largest().ilogb());
    assert_eq!(1023, (-Double::largest()).ilogb());

    assert_eq!(0, "0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(0, "-0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(42, "0x1p+42".parse::<Single>().unwrap().ilogb());
    assert_eq!(-42, "0x1p-42".parse::<Single>().unwrap().ilogb());

    assert_eq!(IEK_INF, Single::INFINITY.ilogb());
    assert_eq!(IEK_INF, (-Single::INFINITY).ilogb());
    assert_eq!(IEK_ZERO, Single::ZERO.ilogb());
    assert_eq!(IEK_ZERO, (-Single::ZERO).ilogb());
    assert_eq!(IEK_NAN, Single::NAN.ilogb());
    assert_eq!(IEK_NAN, Single::snan(None).ilogb());

    assert_eq!(127, Single::largest().ilogb());
    assert_eq!(127, (-Single::largest()).ilogb());

    assert_eq!(-149, Single::SMALLEST.ilogb());
    assert_eq!(-149, (-Single::SMALLEST).ilogb());
    assert_eq!(-126, Single::smallest_normalized().ilogb());
    assert_eq!(-126, (-Single::smallest_normalized()).ilogb());
}

#[test]
fn scalbn() {
    assert!(
        "0x1p+0"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(0),)
    );
    assert!(
        "0x1p+42"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(42),)
    );
    assert!(
        "0x1p-42"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(-42),)
    );

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let snan = Single::snan(None);

    assert!(p_inf.bitwise_eq(p_inf.scalbn(0)));
    assert!(m_inf.bitwise_eq(m_inf.scalbn(0)));
    assert!(p_zero.bitwise_eq(p_zero.scalbn(0)));
    assert!(m_zero.bitwise_eq(m_zero.scalbn(0)));
    assert!(p_qnan.bitwise_eq(p_qnan.scalbn(0)));
    assert!(m_qnan.bitwise_eq(m_qnan.scalbn(0)));
    assert!(!snan.scalbn(0).is_signaling());

    let scalbn_snan = snan.scalbn(1);
    assert!(scalbn_snan.is_nan() && !scalbn_snan.is_signaling());

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));
    let quiet_payload = snan_with_payload.scalbn(1);
    assert!(quiet_payload.is_nan() && !quiet_payload.is_signaling());
    assert_eq!(payload, quiet_payload.to_bits() & ((1 << 51) - 1));

    assert!(p_inf.bitwise_eq("0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(m_inf.bitwise_eq("-0x1p+0".parse::<Single>().unwrap().scalbn(128),));
    assert!(p_inf.bitwise_eq("0x1p+127".parse::<Single>().unwrap().scalbn(1),));
    assert!(p_zero.bitwise_eq("0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!(m_zero.bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-127),));
    assert!(
        "-0x1p-149"
            .parse::<Single>()
            .unwrap()
            .bitwise_eq("-0x1p-127".parse::<Single>().unwrap().scalbn(-22),)
    );
    assert!(p_zero.bitwise_eq("0x1p-126".parse::<Single>().unwrap().scalbn(-24),));

    let smallest_f64 = Double::SMALLEST;
    let neg_smallest_f64 = -Double::SMALLEST;

    let largest_f64 = Double::largest();
    let neg_largest_f64 = -Double::largest();

    let largest_denormal_f64 = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal_f64 = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    assert!(smallest_f64.bitwise_eq("0x1p-1074".parse::<Double>().unwrap().scalbn(0),));
    assert!(neg_smallest_f64.bitwise_eq("-0x1p-1074".parse::<Double>().unwrap().scalbn(0),));

    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2097,),));

    assert!(smallest_f64.scalbn(-2097).is_pos_zero());
    assert!(smallest_f64.scalbn(-2098).is_pos_zero());
    assert!(smallest_f64.scalbn(-2099).is_pos_zero());
    assert!("0x1p+1022".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2096,),));
    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2097,),));
    assert!(smallest_f64.scalbn(2098).is_infinite());
    assert!(smallest_f64.scalbn(2099).is_infinite());

    // Test for integer overflows when adding to exponent.
    assert!(smallest_f64.scalbn(-ExpInt::MAX).is_pos_zero());
    assert!(largest_f64.scalbn(ExpInt::MAX).is_infinite());

    assert!(largest_denormal_f64.bitwise_eq(largest_denormal_f64.scalbn(0),));
    assert!(neg_largest_denormal_f64.bitwise_eq(neg_largest_denormal_f64.scalbn(0),));

    assert!(
        "0x1.ffffffffffffep-1022"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1))
    );
    assert!(
        "-0x1.ffffffffffffep-1021"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(neg_largest_denormal_f64.scalbn(2))
    );

    assert!(
        "0x1.ffffffffffffep+1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1024))
    );
    assert!(largest_denormal_f64.scalbn(-1023).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-1024).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-2048).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(2047).is_infinite());
    assert!(largest_denormal_f64.scalbn(2098).is_infinite());
    assert!(largest_denormal_f64.scalbn(2099).is_infinite());

    assert!(
        "0x1.ffffffffffffep-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1021))
    );
    assert!(
        "0x1.ffffffffffffep-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1022))
    );
    assert!(
        "0x1.ffffffffffffep+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1023))
    );
    assert!(
        "0x1.ffffffffffffep+1023"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(2046))
    );
    assert!("0x1p+974".parse::<Double>().unwrap().bitwise_eq(smallest_f64.scalbn(2048,),));

    let random_denormal_f64 = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap();
    assert!(
        "0x1.c60f120d9f87cp-972"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-1023))
    );
    assert!(
        "0x1.c60f120d9f87cp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-52))
    );
    assert!(
        "0x1.c60f120d9f87cp-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-53))
    );
    assert!(
        "0x1.c60f120d9f87cp+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-51))
    );

    assert!(random_denormal_f64.scalbn(-2097).is_pos_zero());
    assert!(random_denormal_f64.scalbn(-2090).is_pos_zero());

    assert!("-0x1p-1073".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2097),));

    assert!("-0x1p-1024".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2048),));

    assert!("0x1p-1073".parse::<Double>().unwrap().bitwise_eq(largest_f64.scalbn(-2097,),));

    assert!("0x1p-1074".parse::<Double>().unwrap().bitwise_eq(largest_f64.scalbn(-2098,),));
    assert!("-0x1p-1074".parse::<Double>().unwrap().bitwise_eq(neg_largest_f64.scalbn(-2098),));
    assert!(neg_largest_f64.scalbn(-2099).is_neg_zero());
    assert!(largest_f64.scalbn(1).is_infinite());

    assert!(
        "0x1p+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq("0x1p+52".parse::<Double>().unwrap().scalbn(-52),)
    );

    assert!(
        "0x1p-103"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq("0x1p-51".parse::<Double>().unwrap().scalbn(-52),)
    );
}

#[test]
fn frexp() {
    let p_zero = Double::ZERO;
    let m_zero = -Double::ZERO;
    let one = Double::from_f64(1.0);
    let m_one = Double::from_f64(-1.0);

    let largest_denormal = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    let smallest = Double::SMALLEST;
    let neg_smallest = -Double::SMALLEST;

    let largest = Double::largest();
    let neg_largest = -Double::largest();

    let p_inf = Double::INFINITY;
    let m_inf = -Double::INFINITY;

    let p_qnan = Double::NAN;
    let m_qnan = -Double::NAN;
    let snan = Double::snan(None);

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));

    let mut exp = 0;

    let frac = p_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_pos_zero());

    let frac = m_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_neg_zero());

    let frac = one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = m_one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!("-0x1.ffffffffffffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!("-0x1.fffffffffffffp-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = p_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && !frac.is_negative());

    let frac = m_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && frac.is_negative());

    let frac = p_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = m_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = snan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());

    let frac = snan_with_payload.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());
    assert_eq!(payload, frac.to_bits() & ((1 << 51) - 1));

    let frac = "0x0.ffffp-1".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-1, exp);
    assert!("0x1.fffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1p-51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-50, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(52, exp);
    assert!("0x1.c60f120d9f87cp-1".parse::<Double>().unwrap().bitwise_eq(frac));
}
