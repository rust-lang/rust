//! Registering limits:
//! - recursion_limit: there are various parts of the compiler that must impose arbitrary limits
//!   on how deeply they recurse to prevent stack overflow.
//! - move_size_limit
//! - type_length_limit
//! - pattern_complexity_limit
//!
//! Users can override these limits via an attribute on the crate like
//! `#![recursion_limit="22"]`. This pass just looks for those attributes.

use rustc_ast::{self as ast, DUMMY_NODE_ID};
use rustc_attr_ir::target::Target;
use rustc_attr_ir::{Attribute, find_attr};
use rustc_attr_parsing::{AttributeParser, ShouldEmit};
use rustc_feature::Features;
use rustc_session::{Limits, Session};
use rustc_span::{DUMMY_SP, sym};
use rustc_structures::Limit;

pub fn get_limits(krate_attrs: &[ast::Attribute], sess: &Session, features: &Features) -> Limits {
    let attrs = AttributeParser::parse_limited_all(
        sess,
        &krate_attrs,
        Some(&|attr| {
            attr.has_any_name(&[
                sym::recursion_limit,
                sym::move_size_limit,
                sym::type_length_limit,
                sym::pattern_complexity_limit,
            ])
        }),
        Target::Crate,
        DUMMY_SP,
        DUMMY_NODE_ID,
        Some(features),
        // errors are fatal here, but lints aren't.
        // If things aren't fatal we continue, and will parse this again.
        // That makes the same lint trigger again.
        // So, no lints here to avoid duplicates.
        ShouldEmit::EarlyFatal { also_emit_lints: false },
        None,
    );
    let attrs = &attrs;
    Limits {
        recursion_limit: get_recursion_limit(attrs, sess),
        move_size_limit: find_attr!(attrs, MoveSizeLimit { limit } => *limit)
            .unwrap_or(Limit::new(sess.opts.unstable_opts.move_size_limit.unwrap_or(0))),
        type_length_limit: find_attr!(attrs, TypeLengthLimit { limit } => *limit)
            .unwrap_or(Limit::new(2usize.pow(24))),
        pattern_complexity_limit: find_attr!(attrs, PatternComplexityLimit { limit } => *limit)
            .unwrap_or(Limit::unlimited()),
    }
}

fn get_recursion_limit(attrs: &[Attribute], sess: &Session) -> Limit {
    let limit_from_crate = find_attr!(attrs, RecursionLimit { limit } => limit.0).unwrap_or(128);
    Limit::new(
        sess.opts
            .unstable_opts
            .min_recursion_limit
            .map_or(limit_from_crate, |min| min.max(limit_from_crate)),
    )
}
