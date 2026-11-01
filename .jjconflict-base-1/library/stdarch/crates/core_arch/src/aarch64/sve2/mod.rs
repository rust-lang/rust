//! SVE2 intrinsics

#![allow(non_camel_case_types)]

// `generated.rs` has a `super::*` and this import is for that
use super::sve::*;
use crate::intrinsics::*;

#[rustfmt::skip]
mod generated;
#[rustfmt::skip]
#[unstable(feature = "stdarch_aarch64_sve", issue = "145052")]
pub use self::generated::*;

#[cfg(test)]
#[path = "ld_st_tests_aarch64.rs"]
mod ld_st_tests;
