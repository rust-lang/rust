// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::abi::Abi;
use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span, respan};
use syntax::ptr::P;

use block::BlockBuilder;
use fn_decl::FnDeclBuilder;
use generics::GenericsBuilder;
use ident::ToIdent;
use invoke::{Invoke, Identity};
use lifetime::IntoLifetime;

//////////////////////////////////////////////////////////////////////////////

pub struct Method {
    pub sig: ast::MethodSig,
    pub block: Option<P<ast::Block>>,
}

//////////////////////////////////////////////////////////////////////////////

pub struct MethodBuilder<F=Identity> {
    callback: F,
    span: Span,
    abi: Abi,
    generics: ast::Generics,
    unsafety: ast::Unsafety,
    constness: ast::Constness,
    explicit_self: ast::ExplicitSelf,
    fn_decl: P<ast::FnDecl>,
    block: Option<P<ast::Block>>,
}

impl MethodBuilder {
    pub fn new() -> Self {
        MethodBuilder::new_with_callback(Identity)
    }
}

impl<F> MethodBuilder<F>
    where F: Invoke<Method>,
{
    pub fn new_with_callback(callback: F) -> Self {
        MethodBuilder {
            callback: callback,
            span: DUMMY_SP,
            abi: Abi::Rust,
            generics: GenericsBuilder::new().build(),
            unsafety: ast::Unsafety::Normal,
            constness: ast::Constness::NotConst,
            explicit_self: respan(DUMMY_SP, ast::ExplicitSelf_::SelfStatic),
            fn_decl: P(ast::FnDecl {
                inputs: vec![],
                output: ast::FunctionRetTy::NoReturn(DUMMY_SP),
                variadic: false
            }),
            block: None,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn unsafe_(mut self) -> Self {
        self.unsafety = ast::Unsafety::Normal;
        self
    }

    pub fn const_(mut self) -> Self {
        self.constness = ast::Constness::Const;
        self
    }

    pub fn abi(mut self, abi: Abi) -> Self {
        self.abi = abi;
        self
    }

    pub fn with_generics(mut self, generics: ast::Generics) -> Self {
        self.generics = generics;
        self
    }

    pub fn generics(self) -> GenericsBuilder<Self> {
        GenericsBuilder::new_with_callback(self)
    }

    pub fn with_self(mut self, explicit_self: ast::ExplicitSelf) -> Self {
        self.explicit_self = explicit_self;
        self
    }

    pub fn self_(self) -> SelfBuilder<Self> {
        SelfBuilder::new_with_callback(self)
    }

    pub fn with_fn_decl(mut self, fn_decl: P<ast::FnDecl>) -> Self {
        self.fn_decl = fn_decl;
        self
    }

    pub fn fn_decl(self) -> FnDeclBuilder<Self> {
        FnDeclBuilder::new_with_callback(self)
    }

    pub fn with_block(mut self, block: P<ast::Block>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn block(self) -> BlockBuilder<Self> {
        BlockBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        let method_sig = ast::MethodSig {
            unsafety: self.unsafety,
            constness: self.constness,
            abi: self.abi,
            decl: self.fn_decl,
            generics: self.generics,
            explicit_self: self.explicit_self,
        };
        self.callback.invoke(Method {
            sig: method_sig,
            block: self.block,
        })
    }
}

impl<F> Invoke<ast::Generics> for MethodBuilder<F>
    where F: Invoke<Method>,
{
    type Result = Self;

    fn invoke(self, generics: ast::Generics) -> Self {
        self.with_generics(generics)
    }
}

impl<F> Invoke<ast::ExplicitSelf> for MethodBuilder<F>
    where F: Invoke<Method>,
{
    type Result = Self;

    fn invoke(self, explicit_self: ast::ExplicitSelf) -> Self {
        self.with_self(explicit_self)
    }
}

impl<F> Invoke<P<ast::FnDecl>> for MethodBuilder<F>
    where F: Invoke<Method>,
{
    type Result = Self;

    fn invoke(self, fn_decl: P<ast::FnDecl>) -> Self {
        self.with_fn_decl(fn_decl)
    }
}

impl<F> Invoke<P<ast::Block>> for MethodBuilder<F>
    where F: Invoke<Method>,
{
    type Result = Self;

    fn invoke(self, block: P<ast::Block>) -> Self {
        self.with_block(block)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct SelfBuilder<F> {
    callback: F,
    span: Span,
}

impl<F> SelfBuilder<F>
    where F: Invoke<ast::ExplicitSelf>,
{
    pub fn new_with_callback(callback: F) -> Self {
        SelfBuilder {
            callback: callback,
            span: DUMMY_SP,
        }
    }

    pub fn build(self, self_: ast::ExplicitSelf) -> F::Result {
        self.callback.invoke(self_)
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn build_self_(self, self_: ast::ExplicitSelf_) -> F::Result {
        let self_ = respan(self.span, self_);
        self.build(self_)
    }

    pub fn static_(self) -> F::Result {
        self.build_self_(ast::ExplicitSelf_::SelfStatic)
    }

    pub fn value(self) -> F::Result {
        self.build_self_(ast::ExplicitSelf_::SelfValue("self".to_ident()))
    }

    pub fn ref_(self) -> F::Result {
        self.build_self_(ast::ExplicitSelf_::SelfRegion(
            None,
            ast::Mutability::MutImmutable,
            "self".to_ident(),
        ))
    }

    pub fn ref_lifetime<L>(self, lifetime: L) -> F::Result
        where L: IntoLifetime,
    {
        self.build_self_(ast::ExplicitSelf_::SelfRegion(
            Some(lifetime.into_lifetime()),
            ast::Mutability::MutImmutable,
            "self".to_ident(),
        ))
    }

    pub fn ref_mut(self) -> F::Result {
        self.build_self_(ast::ExplicitSelf_::SelfRegion(
            None,
            ast::Mutability::MutMutable,
            "self".to_ident(),
        ))
    }

    pub fn ref_mut_lifetime<L>(self, lifetime: L) -> F::Result
        where L: IntoLifetime,
    {
        self.build_self_(ast::ExplicitSelf_::SelfRegion(
            Some(lifetime.into_lifetime()),
            ast::Mutability::MutMutable,
            "self".to_ident(),
        ))
    }

    /*
    pub fn ty(self) -> TyBuilder<F::Result> {
        TyBuilder::new_with_callback(self)
    }
    */
}

impl<F> Invoke<P<ast::Ty>> for SelfBuilder<F>
    where F: Invoke<ast::ExplicitSelf>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.build_self_(ast::ExplicitSelf_::SelfExplicit(ty, "self".to_ident()))
    }
}
