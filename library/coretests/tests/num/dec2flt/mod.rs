#![allow(overflowing_literals)]

mod decimal;
mod decimal_seq;
mod float;
mod lemire;
mod parse;

// Take a float literal, turn it into a string in various ways (that are all trusted
// to be correct) and see if those strings are parsed back to the value of the literal.
// Requires a *polymorphic literal*, i.e., one that can serve as f64 as well as f32.
macro_rules! test_literal {
    ($x: expr) => {{
        #[cfg(target_has_reliable_f16)]
        let x16: f16 = $x;
        let x32: f32 = $x;
        let x64: f64 = $x;
        let inputs = &[stringify!($x).into(), format!("{:?}", x64), format!("{:e}", x64)];

        for input in inputs {
            assert_eq!(input.parse(), Ok(x64), "failed f64 {input}");
            assert_eq!(input.parse(), Ok(x32), "failed f32 {input}");
            #[cfg(target_has_reliable_f16)]
            assert_eq!(input.parse(), Ok(x16), "failed f16 {input}");

            let neg_input = format!("-{input}");
            assert_eq!(neg_input.parse(), Ok(-x64), "failed f64 {neg_input}");
            assert_eq!(neg_input.parse(), Ok(-x32), "failed f32 {neg_input}");
            #[cfg(target_has_reliable_f16)]
            assert_eq!(neg_input.parse(), Ok(-x16), "failed f16 {neg_input}");
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
    #[cfg(target_has_reliable_f16)]
    assert!(".".parse::<f16>().is_err());
    assert!(".".parse::<f32>().is_err());
    assert!(".".parse::<f64>().is_err());
}

#[test]
fn exponentiated_dot() {
    #[cfg(target_has_reliable_f16)]
    assert!(".e0".parse::<f16>().is_err());
    assert!(".e0".parse::<f32>().is_err());
    assert!(".e0".parse::<f64>().is_err());
}

#[test]
fn lonely_sign() {
    #[cfg(target_has_reliable_f16)]
    assert!("+".parse::<f16>().is_err());
    assert!("-".parse::<f32>().is_err());
    assert!("+".parse::<f64>().is_err());
}

#[test]
fn whitespace() {
    #[cfg(target_has_reliable_f16)]
    assert!("1.0 ".parse::<f16>().is_err());
    assert!(" 1.0".parse::<f32>().is_err());
    assert!("1.0 ".parse::<f64>().is_err());
}

#[test]
fn nan() {
    #[cfg(target_has_reliable_f16)]
    {
        assert!("NaN".parse::<f16>().unwrap().is_nan());
        assert!("-NaN".parse::<f16>().unwrap().is_nan());
    }

    assert!("NaN".parse::<f32>().unwrap().is_nan());
    assert!("-NaN".parse::<f32>().unwrap().is_nan());

    assert!("NaN".parse::<f64>().unwrap().is_nan());
    assert!("-NaN".parse::<f64>().unwrap().is_nan());
}

#[test]
fn inf() {
    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!("inf".parse(), Ok(f16::INFINITY));
        assert_eq!("-inf".parse(), Ok(f16::NEG_INFINITY));
    }

    assert_eq!("inf".parse(), Ok(f32::INFINITY));
    assert_eq!("-inf".parse(), Ok(f32::NEG_INFINITY));

    assert_eq!("inf".parse(), Ok(f64::INFINITY));
    assert_eq!("-inf".parse(), Ok(f64::NEG_INFINITY));
}

#[test]
fn massive_exponent() {
    #[cfg(target_has_reliable_f16)]
    {
        let max = i16::MAX;
        assert_eq!(format!("1e{max}000").parse(), Ok(f16::INFINITY));
        assert_eq!(format!("1e-{max}000").parse(), Ok(0.0f16));
        assert_eq!(format!("1e{max}000").parse(), Ok(f16::INFINITY));
    }

    let max = i32::MAX;
    assert_eq!(format!("1e{max}000").parse(), Ok(f32::INFINITY));
    assert_eq!(format!("1e-{max}000").parse(), Ok(0.0f32));
    assert_eq!(format!("1e{max}000").parse(), Ok(f32::INFINITY));

    let max = i64::MAX;
    assert_eq!(format!("1e{max}000").parse(), Ok(f64::INFINITY));
    assert_eq!(format!("1e-{max}000").parse(), Ok(0.0f64));
    assert_eq!(format!("1e{max}000").parse(), Ok(f64::INFINITY));
}

#[test]
fn long_exponent_cancellation() {
    // Regression test for <https://github.com/rust-lang/rust/issues/163674>.
    // The previous code ignored later digits of the exponent, even though, e.g.,
    // a small fractional part could mean that the correct parsed value would not be +infty.
    let zeros = "0".repeat(65_536);
    // 10^65536 * 10^-655360 = 10^-589824
    let tiny = format!("1{zeros}e-655360");
    // 10^-65536 * 10^655360 = 10^589824
    let huge = format!("0.{}1e655360", &zeros[..65_535]);
    assert_eq!(tiny.parse::<f64>(), Ok(0.0));
    assert_eq!(tiny.parse::<f32>(), Ok(0.0));
    assert_eq!(huge.parse::<f64>(), Ok(f64::INFINITY));
    assert_eq!(huge.parse::<f32>(), Ok(f32::INFINITY));
    assert_eq!(format!("-{tiny}").parse::<f64>().map(f64::to_bits), Ok((-0.0f64).to_bits()));
    assert_eq!(format!("-{tiny}").parse::<f32>().map(f32::to_bits), Ok((-0.0f32).to_bits()));
    assert_eq!(format!("-{huge}").parse::<f64>(), Ok(f64::NEG_INFINITY));
    assert_eq!(format!("-{huge}").parse::<f32>(), Ok(f32::NEG_INFINITY));
}

#[test]
fn long_fractional_decimal_point() {
    // Regression test for <https://github.com/rust-lang/rust/issues/163676>.
    // In this code previously, i32 casts of slice lengths were subtracted.
    // This could overflow which would panic when overflow checks were enabled.
    // Now the subtraction happens in i64, and then is clamped. This test does
    // not use the huge input causing panic referenced in the issue but tests
    // the same arithmetic.
    let short = "0.000000000001000000000000000080";
    let long = format!("{short}{}", "0".repeat(100_000));
    assert_eq!(short.parse::<f64>().map(f64::to_bits), Ok(0x3d719799812dea11));
    assert_eq!(long.parse::<f64>().map(f64::to_bits), Ok(0x3d719799812dea11));
    assert_eq!(long.parse::<f32>(), short.parse::<f32>());

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    {
        use core::num::imp::dec2flt::decimal_seq::final_decimal_point;

        let boundary = 1usize << 31;
        for fractional_count in [boundary - 1, boundary, boundary + 1] {
            assert_eq!(
                final_decimal_point(fractional_count - 11, fractional_count, 0),
                -11,
                "fractional_count={fractional_count}",
            );
        }

        assert_eq!(final_decimal_point(1, boundary, boundary as i64), 1);
        assert_eq!(final_decimal_point(boundary, 0, 1 - boundary as i64), 1);
    }
}
