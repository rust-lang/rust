//! # Feature gates
//!
//! This crate declares the set of past and present unstable features in the compiler.
//! Feature gate checking itself is done in `rustc_ast_passes/src/feature_gate.rs`
//! at the moment.
//!
//! Features are enabled in programs via the crate-level attributes of
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once a feature gate is added,
//! even if it is stabilized or removed, *do not remove it*. Instead, move the
//! symbol to the `accepted` or `removed` modules respectively.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

mod accepted;
mod builtin_attrs;
mod removed;
mod unstable;

#[cfg(test)]
mod tests;

use std::env;
use std::num::NonZero;

use rustc_span::symbol::Symbol;

#[derive(Debug, Clone)]
pub struct Feature {
    pub name: Symbol,
    /// For unstable features: the version the feature was added in.
    /// For accepted features: the version the feature got stabilized in.
    /// For removed features we are inconsistent; sometimes this is the
    /// version it got added, sometimes the version it got removed.
    pub since: &'static str,
    issue: Option<NonZero<u32>>,
}

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // First argument is tracking issue link; second argument is an optional
    // help message, which defaults to "remove this attribute".
    Deprecated(&'static str, Option<&'static str>),
}

#[derive(Clone, Copy, Debug, Hash)]
pub enum UnstableFeatures {
    /// Disallow use of unstable features, as on beta/stable channels.
    Disallow,
    /// Allow use of unstable features, as on nightly.
    Allow,
    /// Errors are bypassed for bootstrapping. This is required any time
    /// during the build that feature-related lints are set to warn or above
    /// because the build turns on warnings-as-errors and uses lots of unstable
    /// features. As a result, this is always required for building Rust itself.
    Cheat,
}

impl UnstableFeatures {
    /// Determines whether this compiler allows unstable options/features,
    /// according to whether it was built as a stable/beta compiler or a nightly
    /// compiler, and taking `RUSTC_BOOTSTRAP` into account.
    #[inline(never)]
    pub fn from_environment() -> Self {
        Self::from_environment_inner(|name| env::var(name))
    }

    /// Unit tests can pass a mock `std::env::var` instead of modifying the real environment.
    fn from_environment_inner(
        env_var: impl Fn(&str) -> Result<String, env::VarError>, // std::env::var
    ) -> Self {
        // If `CFG_DISABLE_UNSTABLE_FEATURES` was true when this compiler was
        // built, it is a stable/beta compiler that forbids unstable features.
        let disable_unstable_features =
            option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some_and(|s| s != "0");
        let default_answer = if disable_unstable_features {
            UnstableFeatures::Disallow
        } else {
            UnstableFeatures::Allow
        };

        // Returns true if the given list of comma-separated crate names
        // contains `CARGO_CRATE_NAME`.
        //
        // This is not actually used by bootstrap; it only exists so that when
        // cargo sees a third-party crate trying to set `RUSTC_BOOTSTRAP=1` in
        // build.rs, it can suggest a somewhat less horrifying alternative.
        //
        // See <https://github.com/rust-lang/rust/pull/77802> for context.
        let includes_current_crate = |names: &str| -> bool {
            let Ok(crate_name) = env_var("CARGO_CRATE_NAME") else { return false };
            // Normalize `-` in crate names to `_`.
            let crate_name = crate_name.replace('-', "_");
            names.replace('-', "_").split(',').any(|name| name == crate_name)
        };

        match env_var("RUSTC_BOOTSTRAP").as_deref() {
            // Force the compiler to act as nightly, even if it's stable.
            Ok("1") => UnstableFeatures::Cheat,
            // Force the compiler to act as stable, even if it's nightly.
            Ok("-1") => UnstableFeatures::Disallow,
            // Force nightly if `RUSTC_BOOTSTRAP` contains the current crate name.
            Ok(names) if includes_current_crate(names) => UnstableFeatures::Cheat,
            _ => default_answer,
        }
    }

    pub fn is_nightly_build(&self) -> bool {
        match *self {
            UnstableFeatures::Allow | UnstableFeatures::Cheat => true,
            UnstableFeatures::Disallow => false,
        }
    }
}

fn find_lang_feature_issue(feature: Symbol) -> Option<NonZero<u32>> {
    // Search in all the feature lists.
    if let Some(f) = UNSTABLE_LANG_FEATURES.iter().find(|f| f.name == feature) {
        return f.issue;
    }
    if let Some(f) = ACCEPTED_LANG_FEATURES.iter().find(|f| f.name == feature) {
        return f.issue;
    }
    if let Some(f) = REMOVED_LANG_FEATURES.iter().find(|f| f.feature.name == feature) {
        return f.feature.issue;
    }
    panic!("feature `{feature}` is not declared anywhere");
}

const fn to_nonzero(n: Option<u32>) -> Option<NonZero<u32>> {
    // Can be replaced with `n.and_then(NonZero::new)` if that is ever usable
    // in const context. Requires https://github.com/rust-lang/rfcs/pull/2632.
    match n {
        None => None,
        Some(n) => NonZero::new(n),
    }
}

pub enum GateIssue {
    Language,
    Library(Option<NonZero<u32>>),
}

pub fn find_feature_issue(feature: Symbol, issue: GateIssue) -> Option<NonZero<u32>> {
    match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    }
}

pub use accepted::ACCEPTED_LANG_FEATURES;
pub use builtin_attrs::{
    AttributeDuplicates, AttributeGate, AttributeSafety, AttributeTemplate, AttributeType,
    BUILTIN_ATTRIBUTE_MAP, BUILTIN_ATTRIBUTES, BuiltinAttribute, GatedCfg, deprecated_attributes,
    encode_cross_crate, find_gated_cfg, is_builtin_attr_name, is_stable_diagnostic_attribute,
    is_valid_for_get_attr,
};
pub use removed::REMOVED_LANG_FEATURES;
pub use unstable::{
    EnabledLangFeature, EnabledLibFeature, Features, INCOMPATIBLE_FEATURES, UNSTABLE_LANG_FEATURES,
};
