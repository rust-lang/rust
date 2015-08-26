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
use syntax::codemap::{DUMMY_SP, Span};
use syntax::ptr::P;

use expr::ExprBuilder;
use invoke::{Invoke, Identity};
use ty::TyBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct Const {
    pub ty: P<ast::Ty>,
    pub expr: Option<P<ast::Expr>>,
}

//////////////////////////////////////////////////////////////////////////////

pub struct ConstBuilder<F=Identity> {
    callback: F,
    span: Span,
    expr: Option<P<ast::Expr>>,
}

impl ConstBuilder {
    pub fn new() -> Self {
        ConstBuilder::new_with_callback(Identity)
    }
}

impl<F> ConstBuilder<F>
    where F: Invoke<Const>,
{
    pub fn new_with_callback(callback: F) -> Self
        where F: Invoke<Const>,
    {
        ConstBuilder {
            callback: callback,
            span: DUMMY_SP,
            expr: None,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_expr(mut self, expr: P<ast::Expr>) -> Self {
        self.expr = Some(expr);
        self
    }

    pub fn expr(self) -> ExprBuilder<Self> {
        ExprBuilder::new_with_callback(self)
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn build(self, ty: P<ast::Ty>) -> F::Result {
        self.callback.invoke(Const {
            ty: ty,
            expr: self.expr,
        })
    }
}

impl<F> Invoke<P<ast::Expr>> for ConstBuilder<F>
    where F: Invoke<Const>,
{
    type Result = Self;

    fn invoke(self, expr: P<ast::Expr>) -> Self {
        self.with_expr(expr)
    }
}

impl<F> Invoke<P<ast::Ty>> for ConstBuilder<F>
    where F: Invoke<Const>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.build(ty)
    }
}
