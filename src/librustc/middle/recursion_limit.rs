// Recursion limit.
//
// There are various parts of the compiler that must impose arbitrary limits
// on how deeply they recurse to prevent stack overflow. Users can override
// this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
// just peeks and looks for that attribute.

use session::Session;
use syntax::ast;

use rustc_data_structures::sync::Once;

pub fn update_limits(sess: &Session, krate: &ast::Crate) {
    update_limit(sess, krate, &sess.recursion_limit, "recursion_limit",
                 "recursion limit", 64);
    update_limit(sess, krate, &sess.type_length_limit, "type_length_limit",
                 "type length limit", 1048576);
}

fn update_limit(sess: &Session, krate: &ast::Crate, limit: &Once<usize>,
                name: &str, description: &str, default: usize) {
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

        span_err!(sess, attr.span, E0296,
                  "malformed {} attribute, expected #![{}=\"N\"]",
                  description, name);
    }
    limit.set(default);
}
