use rustc_apfloat::ieee::{Double, Quad, Single, X87DoubleExtended};
use rustc_apfloat::unpack;
use rustc_apfloat::{Float, Round, Status};

#[test]
fn from_zero_decimal_string() {
    assert_eq!(0.0, "0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-00000.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.00000".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0000.00000".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_single_exponent_string() {
    assert_eq!(0.0, "0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "000.0000e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+000.0000e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-000.0000e+1".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_large_exponent_string() {
    assert_eq!(0.0, "0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "000.0000e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "000.0000e-1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_hexadecimal_string() {
    assert_eq!(0.0, "0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x00000.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x00000.p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_decimal_string() {
    assert_eq!(1.0, "1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, ".5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(-4.0, "-4.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.5, "-1.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e12, "1.25e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e+12, "1.25e+12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e-12, "1.25e-12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.0, "1024.".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.05, "1024.05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.05, ".05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e2, "2.e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e+2, "2.e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e-2, "2.e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e2, "002.05000e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e+2, "002.05000e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e-2, "002.05000e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e12, "002.05000e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e+12, "002.05000e+12".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e-12, "002.05000e-12".parse::<Double>().unwrap().to_f64());

    // These are "carefully selected" to overflow the fast log-base
    // calculations in the implementation.
    assert!("99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("-99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("1e-99999".parse::<Double>().unwrap().is_pos_zero());
    assert!("-1e-99999".parse::<Double>().unwrap().is_neg_zero());

    assert_eq!(2.71828, "2.71828".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_hexadecimal_string() {
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p+0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p-0".parse::<Double>().unwrap().to_f64());

    assert_eq!(2.0, "0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2.0, "0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.5, "0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, "+0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-0x1p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(3.0, "0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(3.0, "0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.75, "0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.75, "+0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.75, "-0x1.8p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000.000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000.000p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2048.0, "0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2048.0, "-0x1000.000p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2048.0, "0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2048.0, "-0x1000p-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(16384.0, "0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p10".parse::<Double>().unwrap().to_f64());

    assert_eq!(16384.0, "0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p+10".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.015625, "0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.015625, "+0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.015625, "-0x10p-10".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0625, "0x1.1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(
        "0x1p-150".parse::<Double>().unwrap().to_f64(),
        "+0x800000000000000001.p-221".parse::<Double>().unwrap().to_f64()
    );
    assert_eq!(
        2251799813685248.5,
        "0x80000000000004000000.010p-28".parse::<Double>().unwrap().to_f64()
    );
}

#[test]
fn to_string() {
    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:1$}", x, width)
        } else {
            format!("{:2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.0E+1", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01E-2", to_string(1.01E-2, 5, 1));
    assert_eq!("0.78539816339744828", to_string(0.78539816339744830961, 0, 3));
    assert_eq!("4.9406564584124654E-324", to_string(4.9406564584124654e-324, 0, 3));
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.7318340000000001E+2", to_string(873.1834, 0, 0));
    assert_eq!("1.7976931348623157E+308", to_string(1.7976931348623157E+308, 0, 0));

    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:#1$}", x, width)
        } else {
            format!("{:#2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.000000e+01", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.0100e+04", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01000e+04", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01000e-02", to_string(1.01E-2, 5, 1));
    assert_eq!("0.78539816339744828", to_string(0.78539816339744830961, 0, 3));
    assert_eq!("4.94065645841246540e-324", to_string(4.9406564584124654e-324, 0, 3));
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.73183400000000010e+02", to_string(873.1834, 0, 0));
    assert_eq!("1.79769313486231570e+308", to_string(1.7976931348623157E+308, 0, 0));
}

#[test]
fn to_integer() {
    let mut is_exact = false;

    assert_eq!(
        Status::OK.and(10),
        "10".parse::<Double>().unwrap().to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(0),
        "-10".parse::<Double>().unwrap().to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(31),
        "32".parse::<Double>().unwrap().to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INEXACT.and(7),
        "7.9".parse::<Double>().unwrap().to_u128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::OK.and(-10),
        "-10".parse::<Double>().unwrap().to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(-16),
        "-17".parse::<Double>().unwrap().to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(15),
        "16".parse::<Double>().unwrap().to_i128_r(5, Round::TowardZero, &mut is_exact,)
    );
    assert!(!is_exact);
}

#[test]
fn convert() {
    let mut loses_info = false;
    let test = "1.0".parse::<Double>().unwrap();
    let test: Single = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f32());
    assert!(!loses_info);

    let mut test = "0x1p-53".parse::<X87DoubleExtended>().unwrap();
    let one = "1.0".parse::<X87DoubleExtended>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let mut test = "0x1p-53".parse::<Quad>().unwrap();
    let one = "1.0".parse::<Quad>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let test = "0xf.fffffffp+28".parse::<X87DoubleExtended>().unwrap();
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(4294967295.0, test.to_f64());
    assert!(!loses_info);

    let test = Single::snan(None);
    let x87_snan = X87DoubleExtended::snan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_snan));
    assert!(!loses_info);

    let test = Single::qnan(None);
    let x87_qnan = X87DoubleExtended::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);

    let test = X87DoubleExtended::snan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_snan));
    assert!(!loses_info);

    let test = X87DoubleExtended::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);
}
