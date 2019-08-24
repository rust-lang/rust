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

pub use active::{Features, INCOMPLETE_FEATURES};
pub use builtin_attrs::{
    AttributeGate, AttributeType, GatedCfg,
    BuiltinAttribute, BUILTIN_ATTRIBUTES, BUILTIN_ATTRIBUTE_MAP,
    deprecated_attributes, is_builtin_attr,  is_builtin_attr_name,
};
pub use check::{
    check_attribute, check_crate, get_features, feature_err, emit_feature_err,
    Stability, GateIssue, UnstableFeatures,
    EXPLAIN_STMT_ATTR_SYNTAX, EXPLAIN_UNSIZED_TUPLE_COERCION,
};
