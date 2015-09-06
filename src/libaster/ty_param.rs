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
use syntax::owned_slice::OwnedSlice;
use syntax::ptr::P;

use ident::ToIdent;
use invoke::{Invoke, Identity};
use lifetime::{IntoLifetime, IntoLifetimeDef, LifetimeDefBuilder};
use name::ToName;
use path::IntoPath;

//////////////////////////////////////////////////////////////////////////////

pub struct TyParamBuilder<F=Identity> {
    callback: F,
    span: Span,
    id: ast::Ident,
    bounds: Vec<ast::TyParamBound>,
    default: Option<P<ast::Ty>>,
}

impl TyParamBuilder {
    pub fn new<I>(id: I) -> Self
        where I: ToIdent,
    {
        TyParamBuilder::new_with_callback(id, Identity)
    }

    pub fn from_ty_param(ty_param: ast::TyParam) -> Self {
        TyParamBuilder::from_ty_param_with_callback(Identity, ty_param)
    }
}

impl<F> TyParamBuilder<F>
    where F: Invoke<ast::TyParam>,
{
    pub fn new_with_callback<I>(id: I, callback: F) -> Self
        where I: ToIdent
    {
        TyParamBuilder {
            callback: callback,
            span: DUMMY_SP,
            id: id.to_ident(),
            bounds: Vec::new(),
            default: None,
        }
    }

    pub fn from_ty_param_with_callback(callback: F, ty_param: ast::TyParam) -> Self {
        TyParamBuilder {
            callback: callback,
            span: ty_param.span,
            id: ty_param.ident,
            bounds: ty_param.bounds.into_vec(),
            default: ty_param.default,
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_default(mut self, ty: P<ast::Ty>) -> Self {
        self.default = Some(ty);
        self
    }

    pub fn with_trait_bound(mut self, trait_ref: ast::PolyTraitRef) -> Self {
        self.bounds.push(ast::TyParamBound::TraitTyParamBound(
            trait_ref,
            ast::TraitBoundModifier::None,
        ));
        self
    }

    pub fn trait_bound<P>(self, path: P) -> PolyTraitRefBuilder<Self>
        where P: IntoPath,
    {
        PolyTraitRefBuilder::new_with_callback(path, self)
    }

    pub fn lifetime_bound<L>(mut self, lifetime: L) -> Self
        where L: IntoLifetime,
    {
        let lifetime = lifetime.into_lifetime();

        self.bounds.push(ast::TyParamBound::RegionTyParamBound(lifetime));
        self
    }

    pub fn build(self) -> F::Result {
        self.callback.invoke(ast::TyParam {
            ident: self.id,
            id: ast::DUMMY_NODE_ID,
            bounds: OwnedSlice::from_vec(self.bounds),
            default: self.default,
            span: self.span,
        })
    }
}

impl<F> Invoke<ast::PolyTraitRef> for TyParamBuilder<F>
    where F: Invoke<ast::TyParam>,
{
    type Result = Self;

    fn invoke(self, trait_ref: ast::PolyTraitRef) -> Self {
        self.with_trait_bound(trait_ref)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct PolyTraitRefBuilder<F> {
    callback: F,
    span: Span,
    trait_ref: ast::TraitRef,
    lifetimes: Vec<ast::LifetimeDef>,
}

impl<F> PolyTraitRefBuilder<F>
    where F: Invoke<ast::PolyTraitRef>,
{
    pub fn new_with_callback<P>(path: P, callback: F) -> Self
        where P: IntoPath,
    {
        let trait_ref = ast::TraitRef {
            path: path.into_path(),
            ref_id: ast::DUMMY_NODE_ID,
        };

        PolyTraitRefBuilder {
            callback: callback,
            span: DUMMY_SP,
            trait_ref: trait_ref,
            lifetimes: Vec::new(),
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_lifetime<L>(mut self, lifetime: L) -> Self
        where L: IntoLifetimeDef,
    {
        self.lifetimes.push(lifetime.into_lifetime_def());
        self
    }

    pub fn lifetime<N>(self, name: N) -> LifetimeDefBuilder<Self>
        where N: ToName,
    {
        LifetimeDefBuilder::new_with_callback(name, self)
    }

    pub fn build(self) -> F::Result {
        self.callback.invoke(ast::PolyTraitRef {
            bound_lifetimes: self.lifetimes,
            trait_ref: self.trait_ref,
            span: self.span,
        })
    }
}

impl<F> Invoke<ast::LifetimeDef> for PolyTraitRefBuilder<F>
    where F: Invoke<ast::PolyTraitRef>,
{
    type Result = Self;

    fn invoke(self, lifetime: ast::LifetimeDef) -> Self {
        self.with_lifetime(lifetime)
    }
}
