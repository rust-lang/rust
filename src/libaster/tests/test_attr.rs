// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::{DUMMY_SP, respan};
use syntax::ptr::P;

use super::super::AstBuilder;

#[test]
fn test_doc() {
    let builder = AstBuilder::new();
    assert_eq!(
        builder.attr().doc("/// doc string"),
        respan(
            DUMMY_SP,
            ast::Attribute_ {
                id: ast::AttrId(0),
                style: ast::AttrOuter,
                value: P(respan(
                    DUMMY_SP,
                    ast::MetaNameValue(
                        builder.interned_string("doc"),
                        (*builder.lit().str("/// doc string")).clone(),
                    ),
                )),
                is_sugared_doc: true,
            }
        )
    );
}
