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
use syntax::codemap::{DUMMY_SP, Span, respan};
use syntax::ptr::P;

use ident::ToIdent;
use invoke::{Invoke, Identity};
use ty::TyBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct FnDeclBuilder<F=Identity> {
    callback: F,
    span: Span,
    args: Vec<ast::Arg>,
    variadic: bool,
}

impl FnDeclBuilder {
    pub fn new() -> FnDeclBuilder {
        FnDeclBuilder::new_with_callback(Identity)
    }
}

impl<F> FnDeclBuilder<F>
    where F: Invoke<P<ast::FnDecl>>,
{
    pub fn new_with_callback(callback: F) -> Self {
        FnDeclBuilder {
            callback: callback,
            span: DUMMY_SP,
            args: Vec::new(),
            variadic: false,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn variadic(mut self) -> Self {
        self.variadic = true;
        self
    }

    pub fn with_arg(mut self, arg: ast::Arg) -> Self {
        self.args.push(arg);
        self
    }

    pub fn with_args<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=ast::Arg>
    {
        self.args.extend(iter);
        self
    }

    pub fn arg<I>(self, id: I) -> ArgBuilder<Self>
        where I: ToIdent,
    {
        ArgBuilder::new_with_callback(id, self)
    }

    pub fn no_return(self) -> F::Result {
        let ret_ty = ast::FunctionRetTy::NoReturn(self.span);
        self.build(ret_ty)
    }

    pub fn default_return(self) -> F::Result {
        let ret_ty = ast::FunctionRetTy::DefaultReturn(self.span);
        self.build(ret_ty)
    }

    pub fn build_return(self, ty: P<ast::Ty>) -> F::Result {
        self.build(ast::FunctionRetTy::Return(ty))
    }

    pub fn return_(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn build(self, output: ast::FunctionRetTy) -> F::Result {
        self.callback.invoke(P(ast::FnDecl {
            inputs: self.args,
            output: output,
            variadic: self.variadic,
        }))
    }
}

impl<F> Invoke<ast::Arg> for FnDeclBuilder<F>
    where F: Invoke<P<ast::FnDecl>>
{
    type Result = Self;

    fn invoke(self, arg: ast::Arg) -> Self {
        self.with_arg(arg)
    }
}

impl<F> Invoke<P<ast::Ty>> for FnDeclBuilder<F>
    where F: Invoke<P<ast::FnDecl>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.build_return(ty)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ArgBuilder<F=Identity> {
    callback: F,
    span: Span,
    id: ast::Ident,
}

impl ArgBuilder {
    pub fn new<I>(id: I) -> Self where I: ToIdent {
        ArgBuilder::new_with_callback(id, Identity)
    }
}

impl<F> ArgBuilder<F>
    where F: Invoke<ast::Arg>,
{
    pub fn new_with_callback<I>(id: I, callback: F) -> ArgBuilder<F>
        where I: ToIdent,
    {
        ArgBuilder {
            callback: callback,
            span: DUMMY_SP,
            id: id.to_ident(),
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn build_ty(self, ty: P<ast::Ty>) -> F::Result {
        let path = respan(self.span, self.id);

        self.callback.invoke(ast::Arg {
            id: ast::DUMMY_NODE_ID,
            ty: ty,
            pat: P(ast::Pat {
                id: ast::DUMMY_NODE_ID,
                node: ast::PatIdent(
                    ast::BindByValue(ast::Mutability::MutImmutable),
                    path,
                    None,
                ),
                span: self.span,
            }),
        })
    }

    pub fn ty(self) -> TyBuilder<ArgTyBuilder<F>> {
        TyBuilder::new_with_callback(ArgTyBuilder(self))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct ArgTyBuilder<F>(ArgBuilder<F>);

impl<F: Invoke<ast::Arg>> Invoke<P<ast::Ty>> for ArgTyBuilder<F>
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.0.build_ty(ty)
    }
}
