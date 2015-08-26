// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::IntoIterator;

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::ptr::P;

use expr::ExprBuilder;
use invoke::{Invoke, Identity};
use stmt::StmtBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct BlockBuilder<F=Identity> {
    callback: F,
    span: Span,
    stmts: Vec<P<ast::Stmt>>,
    block_check_mode: ast::BlockCheckMode,
}

impl BlockBuilder {
    pub fn new() -> Self {
        BlockBuilder::new_with_callback(Identity)
    }
}

impl<F> BlockBuilder<F>
    where F: Invoke<P<ast::Block>>,
{
    pub fn new_with_callback(callback: F) -> Self {
        BlockBuilder {
            callback: callback,
            span: DUMMY_SP,
            stmts: Vec::new(),
            block_check_mode: ast::BlockCheckMode::DefaultBlock,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn unsafe_(mut self) -> Self {
        let source = ast::UnsafeSource::CompilerGenerated;
        self.block_check_mode = ast::BlockCheckMode::UnsafeBlock(source);
        self
    }

    pub fn with_stmts<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Stmt>>
    {
        self.stmts.extend(iter);
        self
    }

    pub fn with_stmt(mut self, stmt: P<ast::Stmt>) -> Self {
        self.stmts.push(stmt);
        self
    }

    pub fn stmt(self) -> StmtBuilder<Self> {
        StmtBuilder::new_with_callback(self)
    }

    pub fn build_expr(self, expr: P<ast::Expr>) -> F::Result {
        self.build_(Some(expr))
    }

    pub fn expr(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.build_(None)
    }

    fn build_(self, expr: Option<P<ast::Expr>>) -> F::Result {
        self.callback.invoke(P(ast::Block {
            stmts: self.stmts,
            expr: expr,
            id: ast::DUMMY_NODE_ID,
            rules: self.block_check_mode,
            span: self.span,
        }))
    }
}

impl<F> Invoke<P<ast::Stmt>> for BlockBuilder<F>
    where F: Invoke<P<ast::Block>>,
{
    type Result = Self;

    fn invoke(self, stmt: P<ast::Stmt>) -> Self {
        self.with_stmt(stmt)
    }
}

impl<F> Invoke<P<ast::Expr>> for BlockBuilder<F>
    where F: Invoke<P<ast::Block>>,
{
    type Result = F::Result;

    fn invoke(self, expr: P<ast::Expr>) -> F::Result {
        self.build_expr(expr)
    }
}
