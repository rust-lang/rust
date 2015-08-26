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
use syntax::codemap::DUMMY_SP;
use syntax::owned_slice::OwnedSlice;

use super::super::AstBuilder;

#[test]
fn test_id() {
    let builder = AstBuilder::new();
    let path = builder.path().id("isize").build();

    assert_eq!(
        path,
        ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: vec![
                ast::PathSegment {
                    identifier: builder.id("isize"),
                    parameters: ast::PathParameters::none(),
                },
            ]
        }
    );
}

#[test]
fn test_single_segment() {
    let builder = AstBuilder::new();
    let path = builder.path()
        .segment("isize").build()
        .build();

    assert_eq!(
        path,
        ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: vec![
                ast::PathSegment {
                    identifier: builder.id("isize"),
                    parameters: ast::PathParameters::none(),
                },
            ]
        }
    );
}

#[test]
fn test_multiple_segments() {
    let builder = AstBuilder::new();
    let path = builder.path().global()
        .id("std")
        .id("thread")
        .id("Thread")
        .build();

    assert_eq!(
        path,
        ast::Path {
            span: DUMMY_SP,
            global: true,
            segments: vec![
                ast::PathSegment {
                    identifier: builder.id("std"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: builder.id("thread"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: builder.id("Thread"),
                    parameters: ast::PathParameters::none(),
                },
            ]
        }
    );
}

#[test]
fn test_option() {
    let builder = AstBuilder::new();
    let path = builder.path().global()
        .id("std")
        .id("option")
        .segment("Option")
            .with_ty(builder.ty().id("isize"))
            .build()
        .build();

    assert_eq!(
        path,
        ast::Path {
            span: DUMMY_SP,
            global: true,
            segments: vec![
                ast::PathSegment {
                    identifier: builder.id("std"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: builder.id("option"),
                    parameters: ast::PathParameters::none(),
                },
                ast::PathSegment {
                    identifier: builder.id("Option"),
                    parameters: ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                        lifetimes: vec![],
                        types: OwnedSlice::from_vec(vec![
                            builder.ty().isize(),
                        ]),
                        bindings: OwnedSlice::empty(),
                    }),
                },
            ]
        }
    );
}

#[test]
fn test_lifetimes() {
    let builder = AstBuilder::new();
    let path = builder.path()
        .segment("Foo")
            .lifetime("'a")
            .build()
        .build();

    assert_eq!(
        path,
        ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: vec![
                ast::PathSegment {
                    identifier: builder.id("Foo"),
                    parameters: ast::AngleBracketedParameters(ast::AngleBracketedParameterData {
                        lifetimes: vec![
                            builder.lifetime("'a"),
                        ],
                        types: OwnedSlice::empty(),
                        bindings: OwnedSlice::empty(),
                    }),
                },
            ]
        }
    );
}
