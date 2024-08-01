//! IEEE 754 floating point compliance tests
//!
//! To understand IEEE 754's requirements on a programming language, one must understand that the
//! requirements of IEEE 754 rest on the total programming environment, and not entirely on any
//! one component. That means the hardware, language, and even libraries are considered part of
//! conforming floating point support in a programming environment.
//!
//! A programming language's duty, accordingly, is:
//!   1. offer access to the hardware where the hardware offers support
//!   2. provide operations that fulfill the remaining requirements of the standard
//!   3. provide the ability to write additional software that can fulfill those requirements
//!
//! This may be fulfilled in any combination that the language sees fit. However, to claim that
//! a language supports IEEE 754 is to suggest that it has fulfilled requirements 1 and 2, without
//! deferring minimum requirements to libraries. This is because support for IEEE 754 is defined
//! as complete support for at least one specified floating point type as an "arithmetic" and
//! "interchange" format, plus specified type conversions to "external character sequences" and
//! integer types.
//!
//! For our purposes,
//! "interchange format"          => f32, f64
//! "arithmetic format"           => f32, f64, and any "soft floats"
//! "external character sequence" => str from any float
//! "integer format"              => {i,u}{8,16,32,64,128}
//!
//! None of these tests are against Rust's own implementation. They are only tests against the
//! standard. That is why they accept wildly diverse inputs or may seem to duplicate other tests.
//! Please consider this carefully when adding, removing, or reorganizing these tests. They are
//! here so that it is clear what tests are required by the standard and what can be changed.

use ::core::str::FromStr;

// IEEE 754 for many tests is applied to specific bit patterns.
// These generally are not applicable to NaN, however.
macro_rules! assert_biteq {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!($lhs.to_bits(), $rhs.to_bits())
    };
}

// ToString uses the default fmt::Display impl without special concerns, and bypasses other parts
// of the formatting infrastructure, which makes it ideal for testing here.
macro_rules! roundtrip {
    ($f:expr => $t:ty) => {
        ($f).to_string().parse::<$t>().unwrap()
    };
}

macro_rules! assert_floats_roundtrip {
    ($f:ident) => {
        assert_biteq!(f32::$f, roundtrip!(f32::$f => f32));
        assert_biteq!(f64::$f, roundtrip!(f64::$f => f64));
    };
    ($f:expr) => {
        assert_biteq!($f as f32, roundtrip!($f => f32));
        assert_biteq!($f as f64, roundtrip!($f => f64));
    }
}

macro_rules! assert_floats_bitne {
    ($lhs:ident, $rhs:ident) => {
        assert_ne!(f32::$lhs.to_bits(), f32::$rhs.to_bits());
        assert_ne!(f64::$lhs.to_bits(), f64::$rhs.to_bits());
    };
    ($lhs:expr, $rhs:expr) => {
        assert_ne!(f32::to_bits($lhs), f32::to_bits($rhs));
        assert_ne!(f64::to_bits($lhs), f64::to_bits($rhs));
    };
}

// We must preserve signs on all numbers. That includes zero.
// -0 and 0 are == normally, so test bit equality.
#[test]
fn preserve_signed_zero() {
    assert_floats_roundtrip!(-0.0);
    assert_floats_roundtrip!(0.0);
    assert_floats_bitne!(0.0, -0.0);
}

#[test]
fn preserve_signed_infinity() {
    assert_floats_roundtrip!(INFINITY);
    assert_floats_roundtrip!(NEG_INFINITY);
    assert_floats_bitne!(INFINITY, NEG_INFINITY);
}

#[test]
fn infinity_to_str() {
    assert!(match f32::INFINITY.to_string().to_lowercase().as_str() {
        "+infinity" | "infinity" => true,
        "+inf" | "inf" => true,
        _ => false,
    });
    assert!(
        match f64::INFINITY.to_string().to_lowercase().as_str() {
            "+infinity" | "infinity" => true,
            "+inf" | "inf" => true,
            _ => false,
        },
        "Infinity must write to a string as some casing of inf or infinity, with an optional +."
    );
}

#[test]
fn neg_infinity_to_str() {
    assert!(match f32::NEG_INFINITY.to_string().to_lowercase().as_str() {
        "-infinity" | "-inf" => true,
        _ => false,
    });
    assert!(
        match f64::NEG_INFINITY.to_string().to_lowercase().as_str() {
            "-infinity" | "-inf" => true,
            _ => false,
        },
        "Negative Infinity must write to a string as some casing of -inf or -infinity"
    )
}

#[test]
fn nan_to_str() {
    assert!(
        match f32::NAN.to_string().to_lowercase().as_str() {
            "nan" | "+nan" | "-nan" => true,
            _ => false,
        },
        "NaNs must write to a string as some casing of nan."
    )
}

// "+"?("inf"|"infinity") in any case => Infinity
#[test]
fn infinity_from_str() {
    assert_biteq!(f32::INFINITY, f32::from_str("infinity").unwrap());
    assert_biteq!(f32::INFINITY, f32::from_str("inf").unwrap());
    assert_biteq!(f32::INFINITY, f32::from_str("+infinity").unwrap());
    assert_biteq!(f32::INFINITY, f32::from_str("+inf").unwrap());
    // yes! this means you are weLcOmE tO mY iNfInItElY tWiStEd MiNd
    assert_biteq!(f32::INFINITY, f32::from_str("+iNfInItY").unwrap());
}

// "-inf"|"-infinity" in any case => Negative Infinity
#[test]
fn neg_infinity_from_str() {
    assert_biteq!(f32::NEG_INFINITY, f32::from_str("-infinity").unwrap());
    assert_biteq!(f32::NEG_INFINITY, f32::from_str("-inf").unwrap());
    assert_biteq!(f32::NEG_INFINITY, f32::from_str("-INF").unwrap());
    assert_biteq!(f32::NEG_INFINITY, f32::from_str("-INFinity").unwrap());
}

// ("+"|"-"")?"s"?"nan" in any case => qNaN
#[test]
fn qnan_from_str() {
    assert!("nan".parse::<f32>().unwrap().is_nan());
    assert!("-nan".parse::<f32>().unwrap().is_nan());
    assert!("+nan".parse::<f32>().unwrap().is_nan());
    assert!("+NAN".parse::<f32>().unwrap().is_nan());
    assert!("-NaN".parse::<f32>().unwrap().is_nan());
}
