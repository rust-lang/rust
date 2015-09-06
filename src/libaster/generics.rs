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
use syntax::owned_slice::OwnedSlice;

use ident::ToIdent;
use invoke::{Invoke, Identity};
use lifetime::{IntoLifetime, IntoLifetimeDef, LifetimeDefBuilder};
use name::ToName;
use path::IntoPath;
use ty_param::TyParamBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct GenericsBuilder<F=Identity> {
    callback: F,
    span: Span,
    lifetimes: Vec<ast::LifetimeDef>,
    ty_params: Vec<ast::TyParam>,
    predicates: Vec<ast::WherePredicate>,
}

impl GenericsBuilder {
    pub fn new() -> Self {
        GenericsBuilder::new_with_callback(Identity)
    }

    pub fn from_generics(generics: ast::Generics) -> Self {
        GenericsBuilder::from_generics_with_callback(generics, Identity)
    }
}

impl<F> GenericsBuilder<F>
    where F: Invoke<ast::Generics>,
{
    pub fn new_with_callback(callback: F) -> Self {
        GenericsBuilder {
            callback: callback,
            span: DUMMY_SP,
            lifetimes: Vec::new(),
            ty_params: Vec::new(),
            predicates: Vec::new(),
        }
    }

    pub fn from_generics_with_callback(generics: ast::Generics, callback: F) -> Self {
        GenericsBuilder {
            callback: callback,
            span: DUMMY_SP,
            lifetimes: generics.lifetimes,
            ty_params: generics.ty_params.into_vec(),
            predicates: generics.where_clause.predicates,
        }
    }

    pub fn with(self, generics: ast::Generics) -> Self {
        self.with_lifetimes(generics.lifetimes.into_iter())
            .with_ty_params(generics.ty_params.move_iter())
            .with_predicates(generics.where_clause.predicates.into_iter())
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_lifetimes<I, L>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=L>,
              L: IntoLifetimeDef,
    {
        let iter = iter.into_iter().map(|lifetime_def| lifetime_def.into_lifetime_def());
        self.lifetimes.extend(iter);
        self
    }

    pub fn with_lifetime_names<I, N>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=N>,
              N: ToName,
    {
        for name in iter {
            self = self.lifetime_name(name);
        }
        self
    }

    pub fn with_lifetime(mut self, lifetime: ast::LifetimeDef) -> Self {
        self.lifetimes.push(lifetime);
        self
    }

    pub fn lifetime_name<N>(self, name: N) -> Self
        where N: ToName,
    {
        self.lifetime(name).build()
    }

    pub fn lifetime<N>(self, name: N) -> LifetimeDefBuilder<Self>
        where N: ToName,
    {
        LifetimeDefBuilder::new_with_callback(name, self)
    }

    pub fn with_ty_params<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=ast::TyParam>,
    {
        self.ty_params.extend(iter);
        self
    }

    pub fn with_ty_param_ids<I, T>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=T>,
              T: ToIdent,
    {
        for id in iter {
            self = self.ty_param_id(id);
        }
        self
    }

    pub fn with_ty_param(mut self, ty_param: ast::TyParam) -> Self {
        self.ty_params.push(ty_param);
        self
    }

    pub fn ty_param_id<I>(self, id: I) -> Self
        where I: ToIdent,
    {
        self.ty_param(id).build()
    }

    pub fn ty_param<I>(self, id: I) -> TyParamBuilder<Self>
        where I: ToIdent,
    {
        let span = self.span;
        TyParamBuilder::new_with_callback(id, self).span(span)
    }

    pub fn with_predicates<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=ast::WherePredicate>,
    {
        self.predicates.extend(iter);
        self
    }

    pub fn with_predicate(mut self, predicate: ast::WherePredicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    pub fn add_lifetime_bound<L>(mut self, lifetime: L) -> Self
        where L: IntoLifetime,
    {
        let lifetime = lifetime.into_lifetime();

        for lifetime_def in self.lifetimes.iter_mut() {
            lifetime_def.bounds.push(lifetime.clone());
        }

        for ty_param in self.ty_params.iter_mut() {
            *ty_param = TyParamBuilder::from_ty_param(ty_param.clone())
                .lifetime_bound(lifetime.clone())
                .build();
        }

        self
    }

    pub fn add_ty_param_bound<P>(mut self, path: P) -> Self
        where P: IntoPath,
    {
        let path = path.into_path();

        for ty_param in self.ty_params.iter_mut() {
            *ty_param = TyParamBuilder::from_ty_param(ty_param.clone())
                .trait_bound(path.clone()).build()
                .build();
        }

        self
    }

    pub fn strip_bounds(self) -> Self {
        self.strip_lifetimes()
            .strip_ty_params()
            .strip_predicates()
    }

    pub fn strip_lifetimes(mut self) -> Self {
        for lifetime in self.lifetimes.iter_mut() {
            lifetime.bounds = vec![];
        }
        self
    }

    pub fn strip_ty_params(mut self) -> Self {
        for ty_param in self.ty_params.iter_mut() {
            ty_param.bounds = OwnedSlice::empty();
        }
        self
    }

    pub fn strip_predicates(mut self) -> Self {
        self.predicates = vec![];
        self
    }

    pub fn build(self) -> F::Result {
        self.callback.invoke(ast::Generics {
            lifetimes: self.lifetimes,
            ty_params: OwnedSlice::from_vec(self.ty_params),
            where_clause: ast::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: self.predicates,
            },
        })
    }
}

impl<F> Invoke<ast::LifetimeDef> for GenericsBuilder<F>
    where F: Invoke<ast::Generics>,
{
    type Result = Self;

    fn invoke(self, lifetime: ast::LifetimeDef) -> Self {
        self.with_lifetime(lifetime)
    }
}

impl<F> Invoke<ast::TyParam> for GenericsBuilder<F>
    where F: Invoke<ast::Generics>,
{
    type Result = Self;

    fn invoke(self, ty_param: ast::TyParam) -> Self {
        self.with_ty_param(ty_param)
    }
}
