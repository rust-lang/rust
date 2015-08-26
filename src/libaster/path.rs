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
use syntax::ptr::P;

use invoke::{Invoke, Identity};

use ident::ToIdent;
use name::ToName;
use ty::TyBuilder;

use lifetime::IntoLifetime;

//////////////////////////////////////////////////////////////////////////////

pub trait IntoPath {
    fn into_path(self) -> ast::Path;
}

impl IntoPath for ast::Path {
    fn into_path(self) -> ast::Path {
        self
    }
}

impl IntoPath for ast::Ident {
    fn into_path(self) -> ast::Path {
        PathBuilder::new().id(self).build()
    }
}

impl<'a> IntoPath for &'a str {
    fn into_path(self) -> ast::Path {
        PathBuilder::new().id(self).build()
    }
}

impl IntoPath for String {
    fn into_path(self) -> ast::Path {
        (&*self).into_path()
    }
}

impl<'a, T> IntoPath for &'a [T] where T: ToIdent {
    fn into_path(self) -> ast::Path {
        PathBuilder::new().ids(self).build()
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct PathBuilder<F=Identity> {
    callback: F,
    span: Span,
    global: bool,
}

impl PathBuilder {
    pub fn new() -> Self {
        PathBuilder::new_with_callback(Identity)
    }
}

impl<F> PathBuilder<F>
    where F: Invoke<ast::Path>,
{
    pub fn new_with_callback(callback: F) -> Self {
        PathBuilder {
            callback: callback,
            span: DUMMY_SP,
            global: false,
        }
    }

    pub fn build(self, path: ast::Path) -> F::Result {
        self.callback.invoke(path)
    }

    /// Update the span to start from this location.
    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn global(mut self) -> Self {
        self.global = true;
        self
    }

    pub fn ids<I, T>(self, ids: I) -> PathSegmentsBuilder<F>
        where I: IntoIterator<Item=T>,
              T: ToIdent,
    {
        let mut ids = ids.into_iter();
        let id = ids.next().expect("passed path with no id");

        self.id(id).ids(ids)
    }

    pub fn id<I>(self, id: I) -> PathSegmentsBuilder<F>
        where I: ToIdent,
    {
        self.segment(id).build()
    }

    pub fn segment<I>(self, id: I)
        -> PathSegmentBuilder<PathSegmentsBuilder<F>>
        where I: ToIdent,
    {
        PathSegmentBuilder::new_with_callback(id, PathSegmentsBuilder {
            callback: self.callback,
            span: self.span,
            global: self.global,
            segments: Vec::new(),
        })
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct PathSegmentsBuilder<F=Identity> {
    callback: F,
    span: Span,
    global: bool,
    segments: Vec<ast::PathSegment>,
}

impl<F> PathSegmentsBuilder<F>
    where F: Invoke<ast::Path>,
{
    pub fn ids<I, T>(mut self, ids: I) -> PathSegmentsBuilder<F>
        where I: IntoIterator<Item=T>,
              T: ToIdent,
    {
        for id in ids {
            self = self.id(id);
        }

        self
    }

    pub fn id<T>(self, id: T) -> PathSegmentsBuilder<F>
        where T: ToIdent,
    {
        self.segment(id).build()
    }

    pub fn segment<T>(self, id: T) -> PathSegmentBuilder<Self>
        where T: ToIdent,
    {
        PathSegmentBuilder::new_with_callback(id, self)
    }

    pub fn build(self) -> F::Result {
        self.callback.invoke(ast::Path {
            span: self.span,
            global: self.global,
            segments: self.segments,
        })
    }
}

impl<F> Invoke<ast::PathSegment> for PathSegmentsBuilder<F> {
    type Result = Self;

    fn invoke(mut self, segment: ast::PathSegment) -> Self {
        self.segments.push(segment);
        self
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct PathSegmentBuilder<F=Identity> {
    callback: F,
    span: Span,
    id: ast::Ident,
    lifetimes: Vec<ast::Lifetime>,
    tys: Vec<P<ast::Ty>>,
    bindings: Vec<P<ast::TypeBinding>>,
}

impl<F> PathSegmentBuilder<F>
    where F: Invoke<ast::PathSegment>,
{
    pub fn new_with_callback<I>(id: I, callback: F) -> Self
        where I: ToIdent,
    {
        PathSegmentBuilder {
            callback: callback,
            span: DUMMY_SP,
            id: id.to_ident(),
            lifetimes: Vec::new(),
            tys: Vec::new(),
            bindings: Vec::new(),
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_generics(self, generics: ast::Generics) -> Self {
        // Strip off the bounds.
        let lifetimes = generics.lifetimes.iter()
            .map(|lifetime_def| lifetime_def.lifetime);

        let tys = generics.ty_params.iter()
            .map(|ty_param| TyBuilder::new().id(ty_param.ident));

        self.with_lifetimes(lifetimes)
            .with_tys(tys)
    }

    pub fn with_lifetimes<I, L>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=L>,
              L: IntoLifetime,
    {
        let iter = iter.into_iter().map(|lifetime| lifetime.into_lifetime());
        self.lifetimes.extend(iter);
        self
    }

    pub fn with_lifetime<L>(mut self, lifetime: L) -> Self
        where L: IntoLifetime,
    {
        self.lifetimes.push(lifetime.into_lifetime());
        self
    }

    pub fn lifetime<N>(self, name: N) -> Self
        where N: ToName,
    {
        let lifetime = ast::Lifetime {
            id: ast::DUMMY_NODE_ID,
            span: self.span,
            name: name.to_name(),
        };
        self.with_lifetime(lifetime)
    }

    pub fn with_tys<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Ty>>,
    {
        self.tys.extend(iter);
        self
    }

    pub fn with_ty(mut self, ty: P<ast::Ty>) -> Self {
        self.tys.push(ty);
        self
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        let data = ast::AngleBracketedParameterData {
            lifetimes: self.lifetimes,
            types: OwnedSlice::from_vec(self.tys),
            bindings: OwnedSlice::from_vec(self.bindings),
        };

        let parameters = ast::PathParameters::AngleBracketedParameters(data);

        self.callback.invoke(ast::PathSegment {
            identifier: self.id,
            parameters: parameters,
        })
    }
}

impl<F> Invoke<P<ast::Ty>> for PathSegmentBuilder<F>
    where F: Invoke<ast::PathSegment>
{
    type Result = Self;

    fn invoke(self, ty: P<ast::Ty>) -> Self {
        self.with_ty(ty)
    }
}
