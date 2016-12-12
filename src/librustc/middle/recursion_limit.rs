// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Recursion limit.
//
// There are various parts of the compiler that must impose arbitrary limits
// on how deeply they recurse to prevent stack overflow. Users can override
// this via an attribute on the crate like `#![recursion_limit="22"]`. This pass
// just peeks and looks for that attribute.

use session::Session;
use syntax::ast;

use std::cell::Cell;

pub fn update_limits(sess: &Session, krate: &ast::Crate) {
    update_limit(sess, krate, &sess.recursion_limit, "recursion_limit",
                 "recursion limit");
    update_limit(sess, krate, &sess.type_length_limit, "type_length_limit",
                 "type length limit");
}

fn update_limit(sess: &Session, krate: &ast::Crate, limit: &Cell<usize>,
                name: &str, description: &str) {
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
}
