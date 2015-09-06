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
use syntax::codemap::{DUMMY_SP, Spanned, respan};
use syntax::ptr::P;

use super::super::AstBuilder;

#[test]
fn test_empty() {
    let builder = AstBuilder::new();
    let struct_def = builder.struct_def().build();

    assert_eq!(
        struct_def,
        P(ast::StructDef {
            fields: vec![],
            ctor_id: None,
        })
    );
}

#[test]
fn test_fields() {
    let builder = AstBuilder::new();
    let struct_def = builder.struct_def()
        .field("x").ty().isize()
        .field("y").ty().isize()
        .build();

    assert_eq!(
        struct_def,
        P(ast::StructDef {
            fields: vec![
                Spanned {
                    span: DUMMY_SP,
                    node: ast::StructField_ {
                        kind: ast::NamedField(
                            builder.id("x"),
                            ast::Inherited,
                        ),
                        id: ast::DUMMY_NODE_ID,
                        ty: builder.ty().isize(),
                        attrs: vec![],
                    },
                },
                Spanned {
                    span: DUMMY_SP,
                    node: ast::StructField_ {
                        kind: ast::NamedField(
                            builder.id("y"),
                            ast::Inherited,
                        ),
                        id: ast::DUMMY_NODE_ID,
                        ty: builder.ty().isize(),
                        attrs: vec![],
                    },
                },
            ],
            ctor_id: None,
        })
    );
}

#[test]
fn test_attrs() {
    let builder = AstBuilder::new();
    let struct_def = builder.struct_def()
        .field("x")
            .attr().doc("/// doc string")
            .attr().automatically_derived()
            .ty().isize()
        .build();

    assert_eq!(
        struct_def,
        P(ast::StructDef {
            fields: vec![
                Spanned {
                    span: DUMMY_SP,
                    node: ast::StructField_ {
                        kind: ast::NamedField(
                            builder.id("x"),
                            ast::Inherited,
                        ),
                        id: ast::DUMMY_NODE_ID,
                        ty: builder.ty().isize(),
                        attrs: vec![
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
                            ),
                            respan(
                                DUMMY_SP,
                                ast::Attribute_ {
                                    id: ast::AttrId(1),
                                    style: ast::AttrOuter,
                                    value: P(respan(
                                        DUMMY_SP,
                                        ast::MetaWord(
                                            builder.interned_string("automatically_derived")),
                                    )),
                                    is_sugared_doc: false,
                                }
                            ),
                        ],
                    },
                },
            ],
            ctor_id: None,
        })
    );
}


#[test]
fn test_with_fields() {
    let builder = AstBuilder::new();
    let struct_def = builder.struct_def()
        .field("x").ty().isize()
        .field("y").ty().isize()
        .build();

    let struct_def2 = builder.struct_def()
        .with_fields(
            vec!["x","y"].iter()
                .map(|f| builder.field(f).ty().isize())
            )
        .build();

    assert_eq!(
        struct_def,
        struct_def2
    );
}
