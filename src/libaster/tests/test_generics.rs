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
use syntax::owned_slice::OwnedSlice;

use super::super::AstBuilder;

#[test]
fn test_empty() {
    let builder = AstBuilder::new();
    let generics = builder.generics().build();

    assert_eq!(
        generics,
        ast::Generics {
            lifetimes: vec![],
            ty_params: OwnedSlice::empty(),
            where_clause: ast::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: vec![],
            },
        }
    );
}

#[test]
fn test_with_ty_params_and_lifetimes() {
    let builder = AstBuilder::new();
    let generics = builder.generics()
        .lifetime("'a").build()
        .lifetime("'b").bound("'a").build()
        .ty_param("T").lifetime_bound("'a").build()
        .build();

    assert_eq!(
        generics,
        ast::Generics {
            lifetimes: vec![
                builder.lifetime_def("'a").build(),
                builder.lifetime_def("'b").bound("'a").build(),
            ],
            ty_params: OwnedSlice::from_vec(vec![
                builder.ty_param("T").lifetime_bound("'a").build(),
            ]),
            where_clause: ast::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: vec![],
            },
        }
    );
}
