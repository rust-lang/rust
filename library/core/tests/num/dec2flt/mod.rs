#![allow(overflowing_literals)]

mod decimal;
mod decimal_seq;
mod float;
mod lemire;
mod parse;
mod slow;

// Take a float literal, turn it into a string in various ways (that are all trusted
// to be correct) and see if those strings are parsed back to the value of the literal.
// Requires a *polymorphic literal*, i.e., one that can serve as f64 as well as f32.
macro_rules! test_literal {
    ($x: expr) => {{
        let x16: f16 = $x;
        let x32: f32 = $x;
        let x64: f64 = $x;
        let inputs = &[stringify!($x).into(), format!("{:?}", x64), format!("{:e}", x64)];

        for input in inputs {
            assert_eq!(input.parse(), Ok(x64), "failed f64 {input}");
            assert_eq!(input.parse(), Ok(x32), "failed f32 {input}");
            assert_eq!(input.parse(), Ok(x16), "failed f16 {input}");

            let neg_input = format!("-{input}");
            assert_eq!(neg_input.parse(), Ok(-x64), "failed f64 {neg_input}");
            assert_eq!(neg_input.parse(), Ok(-x32), "failed f32 {neg_input}");
            assert_eq!(neg_input.parse(), Ok(-x16), "failed f16 {neg_input}");
        }
    }};
}

// #[test]
// fn foo() {
//     use core::num::dec2flt::float::RawFloat;
//     use core::num::dec2flt::parse::parse_number;

//     fn x<F: RawFloat + std::fmt::Display>(r: &str) {
//         let mut s = r.as_bytes();
//         let c = s[0];
//         let negative = c == b'-';
//         if c == b'-' || c == b'+' {
//             s = &s[1..];
//         }
//         let mut num = parse_number(s).unwrap();
//         num.negative = negative;
//         if let Some(value) = num.try_fast_path::<F>() {
//             // return Ok(value);
//             println!("fast path {value}");
//             return;
//         }

//         let q = num.exponent;
//         let w = num.mantissa;

//         println!(
//             "float {r} {q} {w} {q:#066b} {w:#066b} sm10 {} lg10 {} ty {} chk {}",
//             F::SMALLEST_POWER_OF_TEN,
//             F::LARGEST_POWER_OF_TEN,
//             std::any::type_name::<F>(),
//             if w == 0 || q < F::SMALLEST_POWER_OF_TEN as i64 {
//                 "lt small 10"
//             } else if q > F::LARGEST_POWER_OF_TEN as i64 {
//                 "gt big 10"
//             } else {
//                 ""
//             }
//         );
//     }

//     // test_literal2!(1e-20);
//     // test_literal2!(1e-30);
//     // test_literal2!(1e-40);
//     // test_literal2!(1e-50);
//     // test_literal2!(1e-60);
//     // test_literal2!(1e-63);
//     // test_literal2!(1e-64);
//     // test_literal2!(1e-65);
//     // test_literal2!(1e-66);
//     // test_literal2!(1e-70);
//     // test_literal2!(1e-70);
//     // test_literal2!(1e-70);
//     // test_literal2!(1e-70);
//     // test_literal2!(2.225073858507201136057409796709131975934819546351645648023426109724822222021076945516529523908135087914149158913039621106870086438694594645527657207407820621743379988141063267329253552286881372149012981122451451889849057222307285255133155755015914397476397983411801999323962548289017107081850690630666655994938275772572015763062690663332647565300009245888316433037779791869612049497390377829704905051080609940730262937128958950003583799967207254304360284078895771796150945516748243471030702609144621572289880258182545180325707018860872113128079512233426288368622321503775666622503982534335974568884423900265498198385487948292206894721689831099698365846814022854243330660339850886445804001034933970427567186443383770486037861622771738545623065874679014086723327636718749999999999999999999999999999999999999e-308);
//     // test_literal2!(1.175494140627517859246175898662808184331245864732796240031385942718174675986064769972472277004271745681762695312500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-38);
//     // panic!();
// }

// #[test]
// fn foobar() {
//     use core::num::dec2flt::float::RawFloat;
//     panic!(
//         "{} {} {} {}",
//         <f32 as RawFloat>::LARGEST_POWER_OF_TEN,
//         <f32 as RawFloat>::SMALLEST_POWER_OF_TEN,
//         <f64 as RawFloat>::LARGEST_POWER_OF_TEN,
//         <f64 as RawFloat>::SMALLEST_POWER_OF_TEN,
//     )
// }

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
fn stats() {
    // use
    use core::num::dec2flt::float::RawFloat;
    dbg!(
        f16::BITS,
        f16::MANTISSA_BITS,
        f16::MANTISSA_EXPLICIT_BITS,
        f16::EXPONENT_BITS,
        f16::MAX_EXPONENT_FAST_PATH,
        f16::MIN_EXPONENT_FAST_PATH,
        f16::MIN_EXPONENT_ROUND_TO_EVEN,
        f16::MAX_EXPONENT_ROUND_TO_EVEN,
        f16::MINIMUM_EXPONENT,
        f16::MAXIMUM_EXPONENT,
        f16::EXPONENT_BIAS,
        f16::INFINITE_POWER,
        f16::LARGEST_POWER_OF_TEN,
        f16::SMALLEST_POWER_OF_TEN,
        f16::MAX_MANTISSA_FAST_PATH
    );
    dbg!(
        f32::BITS,
        f32::MANTISSA_BITS,
        f32::MANTISSA_EXPLICIT_BITS,
        f32::EXPONENT_BITS,
        f32::MAX_EXPONENT_FAST_PATH,
        f32::MIN_EXPONENT_FAST_PATH,
        f32::MIN_EXPONENT_ROUND_TO_EVEN,
        f32::MAX_EXPONENT_ROUND_TO_EVEN,
        f32::MINIMUM_EXPONENT,
        f32::MAXIMUM_EXPONENT,
        f32::EXPONENT_BIAS,
        f32::INFINITE_POWER,
        f32::LARGEST_POWER_OF_TEN,
        f32::SMALLEST_POWER_OF_TEN,
        f32::MAX_MANTISSA_FAST_PATH
    );
    dbg!(
        f64::BITS,
        f64::MANTISSA_BITS,
        f64::MANTISSA_EXPLICIT_BITS,
        f64::EXPONENT_BITS,
        f64::MAX_EXPONENT_FAST_PATH,
        f64::MIN_EXPONENT_FAST_PATH,
        f64::MIN_EXPONENT_ROUND_TO_EVEN,
        f64::MAX_EXPONENT_ROUND_TO_EVEN,
        f64::MINIMUM_EXPONENT,
        f64::MAXIMUM_EXPONENT,
        f64::EXPONENT_BIAS,
        f64::INFINITE_POWER,
        f64::LARGEST_POWER_OF_TEN,
        f64::SMALLEST_POWER_OF_TEN,
        f64::MAX_MANTISSA_FAST_PATH
    );

    panic!();
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
    assert_eq!(format!("1e{max}000").parse(), Ok(f64::INFINITY));
    assert_eq!(format!("1e-{max}000").parse(), Ok(0.0));
    assert_eq!(format!("1e{max}000").parse(), Ok(f64::INFINITY));
}
