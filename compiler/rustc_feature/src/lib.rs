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
// tidy-alphabetical-end

mod accepted;
mod builtin_attrs;
mod removed;
mod unstable;

#[cfg(test)]
mod tests;

use std::num::NonZero;

use rustc_span::Symbol;

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
    /// This takes into account `RUSTC_BOOTSTRAP`.
    ///
    /// If `krate` is [`Some`], then setting `RUSTC_BOOTSTRAP=krate` will enable the nightly
    /// features. Otherwise, only `RUSTC_BOOTSTRAP=1` will work.
    pub fn from_environment(krate: Option<&str>) -> Self {
        Self::from_environment_value(krate, std::env::var("RUSTC_BOOTSTRAP"))
    }

    /// Avoid unsafe `std::env::set_var()` by allowing tests to inject
    /// `std::env::var("RUSTC_BOOTSTRAP")` with the `env_var_rustc_bootstrap`
    /// arg.
    fn from_environment_value(
        krate: Option<&str>,
        env_var_rustc_bootstrap: Result<String, std::env::VarError>,
    ) -> Self {
        // `true` if this is a feature-staged build, i.e., on the beta or stable channel.
        let disable_unstable_features =
            option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some_and(|s| s != "0");
        // Returns whether `krate` should be counted as unstable
        let is_unstable_crate =
            |var: &str| krate.is_some_and(|name| var.split(',').any(|new_krate| new_krate == name));

        let bootstrap = env_var_rustc_bootstrap.ok();
        if let Some(val) = bootstrap.as_deref() {
            match val {
                val if val == "1" || is_unstable_crate(val) => return UnstableFeatures::Cheat,
                // Hypnotize ourselves so that we think we are a stable compiler and thus don't
                // allow any unstable features.
                "-1" => return UnstableFeatures::Disallow,
                _ => {}
            }
        }

        if disable_unstable_features { UnstableFeatures::Disallow } else { UnstableFeatures::Allow }
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
    BUILTIN_ATTRIBUTE_MAP, BUILTIN_ATTRIBUTES, BuiltinAttribute, GatedCfg, encode_cross_crate,
    find_gated_cfg, is_builtin_attr_name, is_stable_diagnostic_attribute, is_valid_for_get_attr,
};
pub use removed::REMOVED_LANG_FEATURES;
pub use unstable::{
    EnabledLangFeature, EnabledLibFeature, Features, INCOMPATIBLE_FEATURES, UNSTABLE_LANG_FEATURES,
};
