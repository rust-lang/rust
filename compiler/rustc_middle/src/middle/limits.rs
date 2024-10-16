//! Registering limits:
//! * recursion_limit,
//! * move_size_limit, and
//! * type_length_limit
//!
//! There are various parts of the compiler that must impose arbitrary limits
//! on how deeply they recurse to prevent stack overflow. Users can override
//! this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
//! just peeks and looks for that attribute.

use std::num::IntErrorKind;

use rustc_ast::attr::AttributeExt;
use rustc_session::{Limit, Limits, Session};
use rustc_span::symbol::{Symbol, sym};

use crate::error::LimitInvalid;
use crate::query::Providers;

pub fn provide(providers: &mut Providers) {
    providers.limits = |tcx, ()| Limits {
        recursion_limit: get_recursion_limit(tcx.hir().krate_attrs(), tcx.sess),
        move_size_limit: get_limit(
            tcx.hir().krate_attrs(),
            tcx.sess,
            sym::move_size_limit,
            tcx.sess.opts.unstable_opts.move_size_limit.unwrap_or(0),
        ),
        type_length_limit: get_limit(
            tcx.hir().krate_attrs(),
            tcx.sess,
            sym::type_length_limit,
            2usize.pow(24),
        ),
    }
}

pub fn get_recursion_limit(krate_attrs: &[impl AttributeExt], sess: &Session) -> Limit {
    get_limit(krate_attrs, sess, sym::recursion_limit, 128)
}

fn get_limit(
    krate_attrs: &[impl AttributeExt],
    sess: &Session,
    name: Symbol,
    default: usize,
) -> Limit {
    match get_limit_size(krate_attrs, sess, name) {
        Some(size) => Limit::new(size),
        None => Limit::new(default),
    }
}

pub fn get_limit_size(
    krate_attrs: &[impl AttributeExt],
    sess: &Session,
    name: Symbol,
) -> Option<usize> {
    for attr in krate_attrs {
        if !attr.has_name(name) {
            continue;
        }

        if let Some(sym) = attr.value_str() {
            match sym.as_str().parse() {
                Ok(n) => return Some(n),
                Err(e) => {
                    let error_str = match e.kind() {
                        IntErrorKind::PosOverflow => "`limit` is too large",
                        IntErrorKind::Empty => "`limit` must be a non-negative integer",
                        IntErrorKind::InvalidDigit => "not a valid integer",
                        IntErrorKind::NegOverflow => {
                            bug!("`limit` should never negatively overflow")
                        }
                        IntErrorKind::Zero => bug!("zero is a valid `limit`"),
                        kind => bug!("unimplemented IntErrorKind variant: {:?}", kind),
                    };
                    sess.dcx().emit_err(LimitInvalid {
                        span: attr.span(),
                        value_span: attr.value_span().unwrap(),
                        error_str,
                    });
                }
            }
        }
    }
    None
}
