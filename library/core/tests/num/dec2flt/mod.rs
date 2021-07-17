#![allow(overflowing_literals)]

mod float;
mod lemire;
mod parse;

// Take a float literal, turn it into a string in various ways (that are all trusted
// to be correct) and see if those strings are parsed back to the value of the literal.
// Requires a *polymorphic literal*, i.e., one that can serve as f64 as well as f32.
macro_rules! test_literal {
    ($x: expr) => {{
        let x32: f32 = $x;
        let x64: f64 = $x;
        let inputs = &[stringify!($x).into(), format!("{:?}", x64), format!("{:e}", x64)];
        for input in inputs {
            assert_eq!(input.parse(), Ok(x64));
            assert_eq!(input.parse(), Ok(x32));
            let neg_input = &format!("-{}", input);
            assert_eq!(neg_input.parse(), Ok(-x64));
            assert_eq!(neg_input.parse(), Ok(-x32));
        }
    }};
}

#[test]
fn ordinary() {
    test_literal!(1.0);
    test_literal!(3e-5);
    test_literal!(0.1);
    test_literal!(12345.);
    test_literal!(0.9999999);
    test_literal!(2.2250738585072014e-308);
}

#[test]
fn special_code_paths() {
    test_literal!(36893488147419103229.0); // 2^65 - 3, triggers half-to-even with even significand
    test_literal!(101e-33); // Triggers the tricky underflow case in AlgorithmM (for f32)
    test_literal!(1e23); // Triggers AlgorithmR
    test_literal!(2075e23); // Triggers another path through AlgorithmR
    test_literal!(8713e-23); // ... and yet another.
}

#[test]
fn large() {
    test_literal!(1e300);
    test_literal!(123456789.34567e250);
    test_literal!(943794359898089732078308743689303290943794359843568973207830874368930329.);
}

#[test]
fn subnormals() {
    test_literal!(5e-324);
    test_literal!(91e-324);
    test_literal!(1e-322);
    test_literal!(13245643e-320);
    test_literal!(2.22507385851e-308);
    test_literal!(2.1e-308);
    test_literal!(4.9406564584124654e-324);
}

#[test]
fn infinity() {
    test_literal!(1e400);
    test_literal!(1e309);
    test_literal!(2e308);
    test_literal!(1.7976931348624e308);
}

#[test]
fn zero() {
    test_literal!(0.0);
    test_literal!(1e-325);
    test_literal!(1e-326);
    test_literal!(1e-500);
}

#[test]
fn fast_path_correct() {
    // This number triggers the fast path and is handled incorrectly when compiling on
    // x86 without SSE2 (i.e., using the x87 FPU stack).
    test_literal!(1.448997445238699);
}

#[test]
fn lonely_dot() {
    assert!(".".parse::<f32>().is_err());
    assert!(".".parse::<f64>().is_err());
}

#[test]
fn exponentiated_dot() {
    assert!(".e0".parse::<f32>().is_err());
    assert!(".e0".parse::<f64>().is_err());
}

#[test]
fn lonely_sign() {
    assert!("+".parse::<f32>().is_err());
    assert!("-".parse::<f64>().is_err());
}

#[test]
fn whitespace() {
    assert!(" 1.0".parse::<f32>().is_err());
    assert!("1.0 ".parse::<f64>().is_err());
}

#[test]
fn nan() {
    assert!("NaN".parse::<f32>().unwrap().is_nan());
    assert!("NaN".parse::<f64>().unwrap().is_nan());
}

#[test]
fn inf() {
    assert_eq!("inf".parse(), Ok(f64::INFINITY));
    assert_eq!("-inf".parse(), Ok(f64::NEG_INFINITY));
    assert_eq!("inf".parse(), Ok(f32::INFINITY));
    assert_eq!("-inf".parse(), Ok(f32::NEG_INFINITY));
}

#[test]
fn massive_exponent() {
    let max = i64::MAX;
    assert_eq!(format!("1e{}000", max).parse(), Ok(f64::INFINITY));
    assert_eq!(format!("1e-{}000", max).parse(), Ok(0.0));
    assert_eq!(format!("1e{}000", max).parse(), Ok(f64::INFINITY));
}

#[test]
fn borderline_overflow() {
    let mut s = "0.".to_string();
    for _ in 0..375 {
        s.push('3');
    }
    // At the time of this writing, this returns Err(..), but this is a bug that should be fixed.
    // It makes no sense to enshrine that in a test, the important part is that it doesn't panic.
    let _ = s.parse::<f64>();
}
