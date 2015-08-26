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
use syntax::codemap::{DUMMY_SP, Spanned};
use syntax::ptr::P;

use super::super::AstBuilder;

#[test]
fn test_empty_tuple_variant() {
    let builder = AstBuilder::new();
    let variant = builder.variant("A").tuple().build();

    assert_eq!(
        variant,
        P(Spanned {
            span: DUMMY_SP,
            node: ast::Variant_ {
                name: builder.id("A"),
                attrs: vec![],
                kind: ast::TupleVariantKind(vec![]),
                id: ast::DUMMY_NODE_ID,
                disr_expr: None,
                vis: ast::Inherited,
            },
        })
    )
}

#[test]
fn test_tuple_variant() {
    let builder = AstBuilder::new();
    let variant = builder.variant("A").tuple()
        .ty().isize()
        .ty().isize()
        .build();

    assert_eq!(
        variant,
        P(Spanned {
            span: DUMMY_SP,
            node: ast::Variant_ {
                name: builder.id("A"),
                attrs: vec![],
                kind: ast::TupleVariantKind(vec![
                    ast::VariantArg {
                        ty: builder.ty().isize(),
                        id: ast::DUMMY_NODE_ID,
                    },
                    ast::VariantArg {
                        ty: builder.ty().isize(),
                        id: ast::DUMMY_NODE_ID,
                    },
                ]),
                id: ast::DUMMY_NODE_ID,
                disr_expr: None,
                vis: ast::Inherited,
            },
        })
    )
}

#[test]
fn test_struct_variant() {
    let builder = AstBuilder::new();
    let variant = builder.variant("A").struct_()
        .field("a").ty().isize()
        .field("b").ty().isize()
        .build();

    assert_eq!(
        variant,
        P(Spanned {
            span: DUMMY_SP,
            node: ast::Variant_ {
                name: builder.id("A"),
                attrs: vec![],
                kind: ast::StructVariantKind(
                    builder.struct_def()
                        .field("a").ty().isize()
                        .field("b").ty().isize()
                        .build()
                ),
                id: ast::DUMMY_NODE_ID,
                disr_expr: None,
                vis: ast::Inherited,
            },
        })
    )
}
