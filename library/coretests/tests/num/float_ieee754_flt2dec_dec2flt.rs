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
//! "interchange format"          => f16, f32, f64, f128
//! "arithmetic format"           => f16, f32, f64, f128, and any "soft floats"
//! "external character sequence" => str from any float
//! "integer format"              => {i,u}{8,16,32,64,128}
//!
//! None of these tests are against Rust's own implementation. They are only tests against the
//! standard. That is why they accept wildly diverse inputs or may seem to duplicate other tests.
//! Please consider this carefully when adding, removing, or reorganizing these tests. They are
//! here so that it is clear what tests are required by the standard and what can be changed.

use core::fmt;
use core::str::FromStr;

use crate::num::{assert_biteq, float_test};

/// ToString uses the default fmt::Display impl without special concerns, and bypasses other parts
/// of the formatting infrastructure, which makes it ideal for testing here.
#[track_caller]
fn string_roundtrip<T>(x: T) -> T
where
    T: FromStr<Err: fmt::Debug> + fmt::Display,
{
    x.to_string().parse::<T>().unwrap()
}

// FIXME(f128): Tests are disabled while we don't have parsing / printing

// We must preserve signs on all numbers. That includes zero.
// -0 and 0 are == normally, so test bit equality.
float_test! {
    name: preserve_signed_zero,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        let neg0 = flt(-0.0);
        let pos0 = flt(0.0);
        assert_biteq!(neg0, string_roundtrip(neg0));
        assert_biteq!(pos0, string_roundtrip(pos0));
        assert_ne!(neg0.to_bits(), pos0.to_bits());
    }
}

float_test! {
    name: preserve_signed_infinity,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        let neg_inf = Float::NEG_INFINITY;
        let pos_inf = Float::INFINITY;
        assert_biteq!(neg_inf, string_roundtrip(neg_inf));
        assert_biteq!(pos_inf, string_roundtrip(pos_inf));
        assert_ne!(neg_inf.to_bits(), pos_inf.to_bits());
    }
}

float_test! {
    name: infinity_to_str,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        assert!(
            match Float::INFINITY.to_string().to_lowercase().as_str() {
                "+infinity" | "infinity" => true,
                "+inf" | "inf" => true,
                _ => false,
            },
            "Infinity must write to a string as some casing of inf or infinity, with an optional +."
        );
        assert!(
            match Float::NEG_INFINITY.to_string().to_lowercase().as_str() {
                "-infinity" | "-inf" => true,
                _ => false,
            },
            "Negative Infinity must write to a string as some casing of -inf or -infinity"
        );
    }
}

float_test! {
    name: nan_to_str,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        assert!(
            match Float::NAN.to_string().to_lowercase().as_str() {
                "nan" | "+nan" | "-nan" => true,
                _ => false,
            },
            "NaNs must write to a string as some casing of nan."
        )
    }
}

float_test! {
    name: infinity_from_str,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        // "+"?("inf"|"infinity") in any case => Infinity
        assert_biteq!(Float::INFINITY, Float::from_str("infinity").unwrap());
        assert_biteq!(Float::INFINITY, Float::from_str("inf").unwrap());
        assert_biteq!(Float::INFINITY, Float::from_str("+infinity").unwrap());
        assert_biteq!(Float::INFINITY, Float::from_str("+inf").unwrap());
        // yes! this means you are weLcOmE tO mY iNfInItElY tWiStEd MiNd
        assert_biteq!(Float::INFINITY, Float::from_str("+iNfInItY").unwrap());

        // "-inf"|"-infinity" in any case => Negative Infinity
        assert_biteq!(Float::NEG_INFINITY, Float::from_str("-infinity").unwrap());
        assert_biteq!(Float::NEG_INFINITY, Float::from_str("-inf").unwrap());
        assert_biteq!(Float::NEG_INFINITY, Float::from_str("-INF").unwrap());
        assert_biteq!(Float::NEG_INFINITY, Float::from_str("-INFinity").unwrap());

    }
}

float_test! {
    name: qnan_from_str,
    attrs: {
        const: #[cfg(false)],
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(false)],
    },
    test {
        // ("+"|"-"")?"s"?"nan" in any case => qNaN
        assert!("nan".parse::<Float>().unwrap().is_nan());
        assert!("-nan".parse::<Float>().unwrap().is_nan());
        assert!("+nan".parse::<Float>().unwrap().is_nan());
        assert!("+NAN".parse::<Float>().unwrap().is_nan());
        assert!("-NaN".parse::<Float>().unwrap().is_nan());
    }
}
