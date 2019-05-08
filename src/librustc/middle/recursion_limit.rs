// Recursion limit.
//
// There are various parts of the compiler that must impose arbitrary limits
// on how deeply they recurse to prevent stack overflow. Users can override
// this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
// just peeks and looks for that attribute.

use crate::session::Session;
use syntax::ast;
use syntax::symbol::{Symbol, sym};

use rustc_data_structures::sync::Once;

pub fn update_limits(sess: &Session, krate: &ast::Crate) {
    update_limit(krate, &sess.recursion_limit, sym::recursion_limit, 64);
    update_limit(krate, &sess.type_length_limit, sym::type_length_limit, 1048576);
}

fn update_limit(krate: &ast::Crate, limit: &Once<usize>, name: Symbol, default: usize) {
    for attr in &krate.attrs {
        if !attr.check_name(name) {
            continue;
        }

        if let Some(s) = attr.value_str() {
            if let Some(n) = s.as_str().parse().ok() {
                limit.set(n);
                return;
            }
        }
    }
    limit.set(default);
}
