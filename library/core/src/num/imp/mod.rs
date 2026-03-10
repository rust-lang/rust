//! Numeric routines, separate from API.

// These modules are public only for testing.
#[cfg(not(no_fp_fmt_parse))]
pub mod bignum;
#[cfg(not(no_fp_fmt_parse))]
pub mod dec2flt;
#[cfg(not(no_fp_fmt_parse))]
pub mod diy_float;
#[cfg(not(no_fp_fmt_parse))]
pub mod flt2dec;
pub mod fmt;

pub(crate) mod int_bits;
pub(crate) mod int_log10;
pub(crate) mod int_sqrt;
pub(crate) mod libm;
pub(crate) mod overflow_panic;
