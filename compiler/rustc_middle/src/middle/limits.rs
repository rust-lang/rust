//! Registering limits, recursion_limit, type_length_limit and const_eval_limit
//!
//! There are various parts of the compiler that must impose arbitrary limits
//! on how deeply they recurse to prevent stack overflow. Users can override
//! this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
//! just peeks and looks for that attribute.

use crate::bug;
use rustc_ast as ast;
use rustc_data_structures::sync::OnceCell;
use rustc_session::{Limit, Session};
use rustc_span::symbol::{sym, Symbol};

use std::num::IntErrorKind;

pub fn update_limits(sess: &Session, krate: &ast::Crate) {
    update_limit(sess, krate, &sess.recursion_limit, sym::recursion_limit, 128);
    update_limit(sess, krate, &sess.type_length_limit, sym::type_length_limit, 1048576);
    update_limit(sess, krate, &sess.const_eval_limit, sym::const_eval_limit, 1_000_000);
}

fn update_limit(
    sess: &Session,
    krate: &ast::Crate,
    limit: &OnceCell<Limit>,
    name: Symbol,
    default: usize,
) {
    for attr in &krate.attrs {
        if !sess.check_name(attr, name) {
            continue;
        }

        if let Some(s) = attr.value_str() {
            match s.as_str().parse() {
                Ok(n) => {
                    limit.set(Limit::new(n)).unwrap();
                    return;
                }
                Err(e) => {
                    let mut err =
                        sess.struct_span_err(attr.span, "`limit` must be a non-negative integer");

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

                    err.span_label(value_span, error_str);
                    err.emit();
                }
            }
        }
    }
    limit.set(Limit::new(default)).unwrap();
}
