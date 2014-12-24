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
// this via an attribute on the crate like `#![recursion_limit(22)]`. This pass
// just peeks and looks for that attribute.

use session::Session;
use syntax::ast;
use syntax::attr::AttrMetaMethods;
use std::str::FromStr;

pub fn update_recursion_limit(sess: &Session, krate: &ast::Crate) {
    for attr in krate.attrs.iter() {
        if !attr.check_name("recursion_limit") {
            continue;
        }

        if let Some(s) = attr.value_str() {
            if let Some(n) = FromStr::from_str(s.get()) {
                sess.recursion_limit.set(n);
                return;
            }
        }

        sess.span_err(attr.span, "malformed recursion limit attribute, \
                                  expected #![recursion_limit(\"N\")]");
    }
}
