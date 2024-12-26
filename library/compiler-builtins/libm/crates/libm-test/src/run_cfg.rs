//! Configuration for how tests get run.

#![allow(unused)]

use std::collections::BTreeMap;
use std::env;
use std::sync::LazyLock;

use crate::{BaseName, FloatTy, Identifier, op};

pub const EXTENSIVE_ENV: &str = "LIBM_EXTENSIVE_TESTS";

/// Context passed to [`CheckOutput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckCtx {
    /// Allowed ULP deviation
    pub ulp: u32,
    pub fn_ident: Identifier,
    pub base_name: BaseName,
    /// Function name.
    pub fn_name: &'static str,
    /// Return the unsuffixed version of the function name.
    pub base_name_str: &'static str,
    /// Source of truth for tests.
    pub basis: CheckBasis,
}

impl CheckCtx {
    /// Create a new check context, using the default ULP for the function.
    pub fn new(fn_ident: Identifier, basis: CheckBasis) -> Self {
        let mut ret = Self {
            ulp: 0,
            fn_ident,
            fn_name: fn_ident.as_str(),
            base_name: fn_ident.base_name(),
            base_name_str: fn_ident.base_name().as_str(),
            basis,
        };
        ret.ulp = crate::default_ulp(&ret);
        ret
    }
}

/// Possible items to test against
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckBasis {
    /// Check against Musl's math sources.
    Musl,
    /// Check against infinite precision (MPFR).
    Mpfr,
}
