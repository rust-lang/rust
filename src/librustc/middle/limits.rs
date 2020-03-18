//! Registering limits, recursion_limit, type_length_limit and const_eval_limit
//!
//! There are various parts of the compiler that must impose arbitrary limits
//! on how deeply they recurse to prevent stack overflow. Users can override
//! this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
//! just peeks and looks for that attribute.

use crate::session::Session;
use core::num::IntErrorKind;
use rustc::bug;
use rustc_ast::ast;
use rustc_span::symbol::{sym, Symbol};

use rustc_data_structures::sync::Once;

// This is the amount of bytes that need to be left on the stack before increasing the size.
// It must be at least as large as the stack required by any code that does not call
// `ensure_sufficient_stack`.
const RED_ZONE: usize = 100 * 1024; // 100k

// Only the first stack that is pushed, grows exponentially (2^n * STACK_PER_RECURSION) from then
// on. This flag has performance relevant characteristics. Don't set it too high.
const STACK_PER_RECURSION: usize = 1 * 1024 * 1024; // 1MB

/// Grows the stack on demand to prevent stack overflow. Call this in strategic locations
/// to "break up" recursive calls. E.g. almost any call to `visit_expr` or equivalent can benefit
/// from this.
///
/// Should not be sprinkled around carelessly, as it causes a little bit of overhead.
pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(RED_ZONE, STACK_PER_RECURSION, f)
}

pub fn update_limits(sess: &Session, krate: &ast::Crate) {
    update_limit(sess, krate, &sess.recursion_limit, sym::recursion_limit, 128);
    update_limit(sess, krate, &sess.type_length_limit, sym::type_length_limit, 1048576);
    update_limit(sess, krate, &sess.const_eval_limit, sym::const_eval_limit, 1_000_000);
}

fn update_limit(
    sess: &Session,
    krate: &ast::Crate,
    limit: &Once<usize>,
    name: Symbol,
    default: usize,
) {
    for attr in &krate.attrs {
        if !attr.check_name(name) {
            continue;
        }

        if let Some(s) = attr.value_str() {
            match s.as_str().parse() {
                Ok(n) => {
                    limit.set(n);
                    return;
                }
                Err(e) => {
                    let mut err =
                        sess.struct_span_err(attr.span, "`limit` must be a non-negative integer");

                    let value_span = attr
                        .meta()
                        .and_then(|meta| meta.name_value_literal().cloned())
                        .map(|lit| lit.span)
                        .unwrap_or(attr.span);

                    let error_str = match e.kind() {
                        IntErrorKind::Overflow => "`limit` is too large",
                        IntErrorKind::Empty => "`limit` must be a non-negative integer",
                        IntErrorKind::InvalidDigit => "not a valid integer",
                        IntErrorKind::Underflow => bug!("`limit` should never underflow"),
                        IntErrorKind::Zero => bug!("zero is a valid `limit`"),
                        kind => bug!("unimplemented IntErrorKind variant: {:?}", kind),
                    };

                    err.span_label(value_span, error_str);
                    err.emit();
                }
            }
        }
    }
    limit.set(default);
}
