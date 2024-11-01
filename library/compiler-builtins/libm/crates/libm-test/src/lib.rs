pub mod gen;
#[cfg(feature = "test-multiprecision")]
pub mod mpfloat;
mod precision;
mod test_traits;

pub use libm::support::{Float, Int};
pub use precision::{MaybeOverride, SpecialCase, multiprec_allowed_ulp, musl_allowed_ulp};
pub use test_traits::{CheckBasis, CheckCtx, CheckOutput, GenerateInput, Hex, TupleCall};

/// Result type for tests is usually from `anyhow`. Most times there is no success value to
/// propagate.
pub type TestResult<T = (), E = anyhow::Error> = Result<T, E>;

// List of all files present in libm's source
include!(concat!(env!("OUT_DIR"), "/all_files.rs"));

/// Return the unsuffixed version of a function name; e.g. `abs` and `absf` both return `abs`,
/// `lgamma_r` and `lgammaf_r` both return `lgamma_r`.
pub fn base_name(name: &str) -> &str {
    let known_mappings = &[
        ("erff", "erf"),
        ("erf", "erf"),
        ("lgammaf_r", "lgamma_r"),
        ("modff", "modf"),
        ("modf", "modf"),
    ];

    match known_mappings.iter().find(|known| known.0 == name) {
        Some(found) => found.1,
        None => name
            .strip_suffix("f")
            .or_else(|| name.strip_suffix("f16"))
            .or_else(|| name.strip_suffix("f128"))
            .unwrap_or(name),
    }
}

/// True if `EMULATED` is set and nonempty. Used to determine how many iterations to run.
pub const fn emulated() -> bool {
    match option_env!("EMULATED") {
        Some(s) if s.is_empty() => false,
        None => false,
        Some(_) => true,
    }
}
