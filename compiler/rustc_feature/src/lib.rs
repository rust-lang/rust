//! # Feature gates
//!
//! This crate declares the set of past and present unstable features in the compiler.
//! Feature gate checking itself is done in `librustc_ast_passes/feature_gate.rs`
//! at the moment.
//!
//! Features are enabled in programs via the crate-level attributes of
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once a feature gate is added,
//! even if it is stabilized or removed, *do not remove it*. Instead, move the
//! symbol to the `accepted` or `removed` modules respectively.

#![feature(once_cell)]

mod accepted;
mod active;
mod builtin_attrs;
mod removed;

use rustc_span::{edition::Edition, symbol::Symbol, Span};
use std::fmt;
use std::num::NonZeroU32;

#[derive(Clone, Copy)]
pub enum State {
    Accepted,
    Active { set: fn(&mut Features, Span) },
    Removed { reason: Option<&'static str> },
    Stabilized { reason: Option<&'static str> },
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Accepted { .. } => write!(f, "accepted"),
            State::Active { .. } => write!(f, "active"),
            State::Removed { .. } => write!(f, "removed"),
            State::Stabilized { .. } => write!(f, "stabilized"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub state: State,
    pub name: Symbol,
    pub since: &'static str,
    issue: Option<u32>, // FIXME: once #58732 is done make this an Option<NonZeroU32>
    pub edition: Option<Edition>,
    description: &'static str,
}

impl Feature {
    fn issue(&self) -> Option<NonZeroU32> {
        self.issue.and_then(NonZeroU32::new)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // First argument is tracking issue link; second argument is an optional
    // help message, which defaults to "remove this attribute".
    Deprecated(&'static str, Option<&'static str>),
}

#[derive(Clone, Copy, Hash)]
pub enum UnstableFeatures {
    /// Hard errors for unstable features are active, as on beta/stable channels.
    Disallow,
    /// Allow features to be activated, as on nightly.
    Allow,
    /// Errors are bypassed for bootstrapping. This is required any time
    /// during the build that feature-related lints are set to warn or above
    /// because the build turns on warnings-as-errors and uses lots of unstable
    /// features. As a result, this is always required for building Rust itself.
    Cheat,
}

impl UnstableFeatures {
    pub fn from_environment() -> UnstableFeatures {
        // `true` if this is a feature-staged build, i.e., on the beta or stable channel.
        let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
        // `true` if we should enable unstable features for bootstrapping.
        let bootstrap = std::env::var("RUSTC_BOOTSTRAP").is_ok();
        match (disable_unstable_features, bootstrap) {
            (_, true) => UnstableFeatures::Cheat,
            (true, _) => UnstableFeatures::Disallow,
            (false, _) => UnstableFeatures::Allow,
        }
    }

    pub fn is_nightly_build(&self) -> bool {
        match *self {
            UnstableFeatures::Allow | UnstableFeatures::Cheat => true,
            UnstableFeatures::Disallow => false,
        }
    }
}

fn find_lang_feature_issue(feature: Symbol) -> Option<NonZeroU32> {
    if let Some(info) = ACTIVE_FEATURES.iter().find(|t| t.name == feature) {
        // FIXME (#28244): enforce that active features have issue numbers
        // assert!(info.issue().is_some())
        info.issue()
    } else {
        // search in Accepted, Removed, or Stable Removed features
        let found = ACCEPTED_FEATURES
            .iter()
            .chain(REMOVED_FEATURES)
            .chain(STABLE_REMOVED_FEATURES)
            .find(|t| t.name == feature);
        match found {
            Some(found) => found.issue(),
            None => panic!("feature `{}` is not declared anywhere", feature),
        }
    }
}

pub enum GateIssue {
    Language,
    Library(Option<NonZeroU32>),
}

pub fn find_feature_issue(feature: Symbol, issue: GateIssue) -> Option<NonZeroU32> {
    match issue {
        GateIssue::Language => find_lang_feature_issue(feature),
        GateIssue::Library(lib) => lib,
    }
}

pub use accepted::ACCEPTED_FEATURES;
pub use active::{Features, ACTIVE_FEATURES, INCOMPATIBLE_FEATURES, INCOMPLETE_FEATURES};
pub use builtin_attrs::{
    deprecated_attributes, find_gated_cfg, is_builtin_attr_name, AttributeGate, AttributeTemplate,
    AttributeType, BuiltinAttribute, GatedCfg, BUILTIN_ATTRIBUTES, BUILTIN_ATTRIBUTE_MAP,
};
pub use removed::{REMOVED_FEATURES, STABLE_REMOVED_FEATURES};
