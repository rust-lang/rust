// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Aster is a syntax ast builder.

#![crate_name = "aster"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![staged_api]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/")]

#![feature(rustc_private)]
#![feature(staged_api)]

extern crate syntax;

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::parse::token;

pub mod attr;
pub mod block;
pub mod constant;
pub mod expr;
pub mod fn_decl;
pub mod generics;
pub mod ident;
pub mod invoke;
pub mod item;
pub mod lifetime;
pub mod lit;
pub mod mac;
pub mod method;
pub mod name;
pub mod pat;
pub mod path;
pub mod qpath;
pub mod stmt;
pub mod str;
pub mod struct_def;
pub mod ty;
pub mod ty_param;
pub mod variant;

#[cfg(test)]
mod tests;

//////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone)]
pub struct AstBuilder {
    span: Span,
}

impl AstBuilder {
    pub fn new() -> AstBuilder {
        AstBuilder {
            span: DUMMY_SP,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn interned_string<S>(&self, s: S) -> token::InternedString
        where S: str::ToInternedString
    {
        s.to_interned_string()
    }

    pub fn id<I>(&self, id: I) -> ast::Ident
        where I: ident::ToIdent
    {
        id.to_ident()
    }

    pub fn name<N>(&self, name: N) -> ast::Name
        where N: name::ToName
    {
        name.to_name()
    }

    pub fn lifetime<L>(&self, lifetime: L) -> ast::Lifetime
        where L: lifetime::IntoLifetime
    {
        lifetime.into_lifetime()
    }

    pub fn attr(&self) -> attr::AttrBuilder {
        attr::AttrBuilder::new()
    }

    pub fn path(&self) -> path::PathBuilder {
        path::PathBuilder::new()
    }

    pub fn ty(&self) -> ty::TyBuilder {
        ty::TyBuilder::new().span(self.span)
    }

    pub fn lifetime_def<N>(&self, name: N) -> lifetime::LifetimeDefBuilder
        where N: name::ToName,
    {
        lifetime::LifetimeDefBuilder::new(name)
    }

    pub fn ty_param<I>(&self, id: I) -> ty_param::TyParamBuilder
        where I: ident::ToIdent,
    {
        ty_param::TyParamBuilder::new(id).span(self.span)
    }

    pub fn from_ty_param(&self, ty_param: ast::TyParam) -> ty_param::TyParamBuilder {
        ty_param::TyParamBuilder::from_ty_param(ty_param)
    }

    pub fn generics(&self) -> generics::GenericsBuilder {
        generics::GenericsBuilder::new().span(self.span)
    }

    pub fn from_generics(&self, generics: ast::Generics) -> generics::GenericsBuilder {
        generics::GenericsBuilder::from_generics(generics).span(self.span)
    }

    pub fn lit(&self) -> lit::LitBuilder {
        lit::LitBuilder::new().span(self.span)
    }

    pub fn expr(&self) -> expr::ExprBuilder {
        expr::ExprBuilder::new().span(self.span)
    }

    pub fn stmt(&self) -> stmt::StmtBuilder {
        stmt::StmtBuilder::new().span(self.span)
    }

    pub fn block(&self) -> block::BlockBuilder {
        block::BlockBuilder::new().span(self.span)
    }

    pub fn pat(&self) -> pat::PatBuilder {
        pat::PatBuilder::new().span(self.span)
    }

    pub fn fn_decl(&self) -> fn_decl::FnDeclBuilder {
        fn_decl::FnDeclBuilder::new().span(self.span)
    }

    pub fn method(&self) -> method::MethodBuilder {
        method::MethodBuilder::new().span(self.span)
    }

    pub fn arg<I>(&self, id: I) -> fn_decl::ArgBuilder
        where I: ident::ToIdent,
    {
        fn_decl::ArgBuilder::new(id).span(self.span)
    }

    pub fn struct_def(&self) -> struct_def::StructDefBuilder {
        struct_def::StructDefBuilder::new().span(self.span)
    }

    pub fn variant<T>(&self, id: T) -> variant::VariantBuilder
        where T: ident::ToIdent,
    {
        variant::VariantBuilder::new(id).span(self.span)
    }

    pub fn field<T>(&self, id: T) -> struct_def::StructFieldBuilder
        where T: ident::ToIdent,
    {
        struct_def::StructFieldBuilder::named(id).span(self.span)
    }

    pub fn item(&self) -> item::ItemBuilder {
        item::ItemBuilder::new().span(self.span)
    }

    pub fn const_(&self) -> constant::ConstBuilder {
        constant::ConstBuilder::new().span(self.span)
    }
}
