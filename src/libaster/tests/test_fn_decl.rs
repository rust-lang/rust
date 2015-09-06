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
use syntax::ptr::P;

use super::super::AstBuilder;

#[test]
fn test_no_args_return_isize() {
    let builder = AstBuilder::new();
    let fn_decl = builder.fn_decl().return_().isize();

    assert_eq!(
        fn_decl,
        P(ast::FnDecl {
            inputs: vec![],
            output: ast::Return(builder.ty().isize()),
            variadic: false,
        })
    );
}

#[test]
fn test_args_return_isize() {
    let builder = AstBuilder::new();
    let fn_decl = builder.fn_decl()
        .arg("x").ty().isize()
        .arg("y").ty().isize()
        .return_().isize();

    assert_eq!(
        fn_decl,
        P(ast::FnDecl {
            inputs: vec![
                ast::Arg {
                    ty: builder.ty().isize(),
                    pat: builder.pat().id("x"),
                    id: ast::DUMMY_NODE_ID,
                },
                ast::Arg {
                    ty: builder.ty().isize(),
                    pat: builder.pat().id("y"),
                    id: ast::DUMMY_NODE_ID,
                },
            ],
            output: ast::Return(builder.ty().isize()),
            variadic: false,
        })
    );
}

#[test]
fn test_no_return() {
    let builder = AstBuilder::new();
    let fn_decl = builder.fn_decl().no_return();

    assert_eq!(
        fn_decl,
        P(ast::FnDecl {
            inputs: vec![],
            output: ast::NoReturn(DUMMY_SP),
            variadic: false,
        })
    );
}

#[test]
fn test_default_return() {
    let builder = AstBuilder::new();
    let fn_decl = builder.fn_decl().default_return();

    assert_eq!(
        fn_decl,
        P(ast::FnDecl {
            inputs: vec![],
            output: ast::DefaultReturn(DUMMY_SP),
            variadic: false,
        })
    );
}
