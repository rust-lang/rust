//! Registering limits:
//! * recursion_limit,
//! * move_size_limit, and
//! * type_length_limit
//!
//! There are various parts of the compiler that must impose arbitrary limits
//! on how deeply they recurse to prevent stack overflow. Users can override
//! this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
//! just peeks and looks for that attribute.

use crate::bug;
use crate::error::LimitInvalid;
use crate::query::Providers;
use rustc_ast::Attribute;
use rustc_session::Session;
use rustc_session::{Limit, Limits};
use rustc_span::symbol::{sym, Symbol};

use std::num::IntErrorKind;

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
            1048576,
        ),
    }
}

pub fn get_recursion_limit(krate_attrs: &[Attribute], sess: &Session) -> Limit {
    get_limit(krate_attrs, sess, sym::recursion_limit, 128)
}

fn get_limit(krate_attrs: &[Attribute], sess: &Session, name: Symbol, default: usize) -> Limit {
    for attr in krate_attrs {
        if !attr.has_name(name) {
            continue;
        }

        if let Some(s) = attr.value_str() {
            match s.as_str().parse() {
                Ok(n) => return Limit::new(n),
                Err(e) => {
                    let value_span = attr
                        .meta()
                        .and_then(|meta| meta.name_value_literal_span())
                        .unwrap_or(attr.span);

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
                    sess.emit_err(LimitInvalid { span: attr.span, value_span, error_str });
                }
            }
        }
    }
    return Limit::new(default);
}
