//! # Feature gating
//!
//! This module implements the gating necessary for preventing certain compiler
//! features from being used by default. This module will crawl a pre-expanded
//! AST to ensure that there are no features which are used that are not
//! enabled.
//!
//! Features are enabled in programs via the crate-level attributes of
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once code for detection of feature
//! gate usage is added, *do not remove it again* even once the feature
//! becomes stable.

mod accepted;
mod removed;
mod active;
mod builtin_attrs;
mod check;

use std::fmt;
use crate::{edition::Edition, symbol::Symbol};
use syntax_pos::Span;

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
    state: State,
    name: Symbol,
    since: &'static str,
    issue: Option<u32>,
    edition: Option<Edition>,
    description: &'static str,
}

pub use active::{Features, INCOMPLETE_FEATURES};
pub use builtin_attrs::{
    AttributeGate, AttributeType, GatedCfg,
    BuiltinAttribute, BUILTIN_ATTRIBUTES, BUILTIN_ATTRIBUTE_MAP,
    deprecated_attributes, is_builtin_attr,  is_builtin_attr_name,
};
pub use check::{
    check_crate, check_attribute, get_features, feature_err, emit_feature_err,
    Stability, GateIssue, UnstableFeatures,
    EXPLAIN_STMT_ATTR_SYNTAX, EXPLAIN_UNSIZED_TUPLE_COERCION,
};
