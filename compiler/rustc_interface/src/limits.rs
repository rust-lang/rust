//! Registering limits:
//! - recursion_limit: there are various parts of the compiler that must impose arbitrary limits
//!   on how deeply they recurse to prevent stack overflow.
//! - move_size_limit
//! - type_length_limit
//! - pattern_complexity_limit
//!
//! Users can override these limits via an attribute on the crate like
//! `#![recursion_limit="22"]`. This pass just looks for those attributes.

use rustc_hir::attrs::AttributeKind;
use rustc_hir::limit::Limit;
use rustc_hir::{Attribute, find_attr};
use rustc_middle::query::Providers;
use rustc_session::Limits;

pub(crate) fn provide(providers: &mut Providers) {
    providers.limits = |tcx, ()| {
        let attrs = tcx.hir_krate_attrs();
        Limits {
            recursion_limit: get_recursion_limit(tcx.hir_krate_attrs()),
            move_size_limit:
                find_attr!(attrs, AttributeKind::MoveSizeLimit { limit, .. } => *limit)
                    .unwrap_or(Limit::new(tcx.sess.opts.unstable_opts.move_size_limit.unwrap_or(0))),
            type_length_limit:
                find_attr!(attrs, AttributeKind::TypeLengthLimit { limit, .. } => *limit)
                    .unwrap_or(Limit::new(2usize.pow(24))),
            pattern_complexity_limit:
                find_attr!(attrs, AttributeKind::PatternComplexityLimit { limit, .. } => *limit)
                    .unwrap_or(Limit::unlimited()),
        }
    }
}

// This one is separate because it must be read prior to macro expansion.
pub(crate) fn get_recursion_limit(attrs: &[Attribute]) -> Limit {
    find_attr!(attrs, AttributeKind::RecursionLimit { limit, .. } => *limit)
        .unwrap_or(Limit::new(128))
}
