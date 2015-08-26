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
fn test_let() {
    let builder = AstBuilder::new();

    assert_eq!(
        builder.stmt()
            .let_().id("x").build(),
        P(respan(
            DUMMY_SP,
            ast::StmtDecl(
                P(respan(
                    DUMMY_SP,
                    ast::DeclLocal(P(ast::Local {
                        pat: builder.pat().id("x"),
                        ty: None,
                        init: None,
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                    })),
                )),
                ast::DUMMY_NODE_ID,
            ),
        ))
    );

    assert_eq!(
        builder.stmt()
            .let_().id("x").ty().i8().build(),
        P(respan(
            DUMMY_SP,
            ast::StmtDecl(
                P(respan(
                    DUMMY_SP,
                    ast::DeclLocal(P(ast::Local {
                        pat: builder.pat().id("x"),
                        ty: Some(builder.ty().i8()),
                        init: None,
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                    })),
                )),
                ast::DUMMY_NODE_ID,
            ),
        ))
    );

    assert_eq!(
        builder.stmt()
            .let_().id("x").expr().i8(5),
        P(respan(
            DUMMY_SP,
            ast::StmtDecl(
                P(respan(
                    DUMMY_SP,
                    ast::DeclLocal(P(ast::Local {
                        pat: builder.pat().id("x"),
                        ty: None,
                        init: Some(builder.expr().i8(5)),
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                    })),
                )),
                ast::DUMMY_NODE_ID,
            ),
        ))
    );

    assert_eq!(
        builder.stmt()
            .let_().id("x").ty().i8().expr().i8(5),
        P(respan(
            DUMMY_SP,
            ast::StmtDecl(
                P(respan(
                    DUMMY_SP,
                    ast::DeclLocal(P(ast::Local {
                        pat: builder.pat().id("x"),
                        ty: Some(builder.ty().i8()),
                        init: Some(builder.expr().i8(5)),
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                    })),
                )),
                ast::DUMMY_NODE_ID,
            ),
        ))
    );

    assert_eq!(
        builder.stmt().let_()
            .tuple()
                .pat().id("x")
                .pat().id("y")
                .build()
            .expr().tuple()
                .expr().u8(0)
                .expr().u16(1)
                .build(),
        P(respan(
            DUMMY_SP,
            ast::StmtDecl(
                P(respan(
                    DUMMY_SP,
                    ast::DeclLocal(P(ast::Local {
                        pat: builder.pat().tuple()
                            .pat().id("x")
                            .pat().id("y")
                            .build(),
                        ty: None,
                        init: Some(
                            builder.expr().tuple()
                                .expr().u8(0)
                                .expr().u16(1)
                                .build()
                        ),
                        id: ast::DUMMY_NODE_ID,
                        span: DUMMY_SP,
                    })),
                )),
                ast::DUMMY_NODE_ID,
            ),
        ))
    );
}
