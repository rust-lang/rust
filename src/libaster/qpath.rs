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

use invoke::{Invoke, Identity};

use ident::ToIdent;
use path::{PathBuilder, PathSegmentBuilder};
use ty::TyBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct QPathBuilder<F=Identity> {
    callback: F,
    span: Span,
}

impl QPathBuilder {
    pub fn new() -> Self {
        QPathBuilder::new_with_callback(Identity)
    }
}

impl<F> QPathBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    /// Construct a `QPathBuilder` that will call the `callback` with a constructed `ast::QSelf`
    /// and `ast::Path`.
    pub fn new_with_callback(callback: F) -> Self {
        QPathBuilder {
            callback: callback,
            span: DUMMY_SP,
        }
    }

    /// Update the span to start from this location.
    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    /// Build a qualified path first by starting with a type builder.
    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    /// Build a qualified path with a concrete type and path.
    pub fn build(self, qself: ast::QSelf, path: ast::Path) -> F::Result {
        self.callback.invoke((qself, path))
    }
}

impl<F> Invoke<P<ast::Ty>> for QPathBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    type Result = QPathTyBuilder<F>;

    fn invoke(self, ty: P<ast::Ty>) -> QPathTyBuilder<F> {
        QPathTyBuilder {
            builder: self,
            ty: ty,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct QPathTyBuilder<F> {
    builder: QPathBuilder<F>,
    ty: P<ast::Ty>,
}

impl<F> QPathTyBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    /// Build a qualified path with a path builder.
    pub fn as_(self) -> PathBuilder<Self> {
        PathBuilder::new_with_callback(self)
    }

    pub fn id<T>(self, id: T) -> F::Result
        where T: ToIdent,
    {
        let path = ast::Path {
            span: self.builder.span,
            global: false,
            segments: vec![],
        };
        self.as_().build(path).id(id)
    }

    pub fn segment<T>(self, id: T) -> PathSegmentBuilder<QPathQSelfBuilder<F>>
        where T: ToIdent,
    {
        let path = ast::Path {
            span: self.builder.span,
            global: false,
            segments: vec![],
        };
        self.as_().build(path).segment(id)
    }
}

impl<F> Invoke<ast::Path> for QPathTyBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    type Result = QPathQSelfBuilder<F>;

    fn invoke(self, path: ast::Path) -> QPathQSelfBuilder<F> {
        QPathQSelfBuilder {
            builder: self.builder,
            qself: ast::QSelf {
                ty: self.ty,
                position: path.segments.len(),
            },
            path: path,
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct QPathQSelfBuilder<F> {
    builder: QPathBuilder<F>,
    qself: ast::QSelf,
    path: ast::Path,
}

impl<F> QPathQSelfBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    pub fn id<T>(self, id: T) -> F::Result
        where T: ToIdent,
    {
        self.segment(id).build()
    }

    pub fn segment<T>(self, id: T) -> PathSegmentBuilder<QPathQSelfBuilder<F>>
        where T: ToIdent,
    {
        PathSegmentBuilder::new_with_callback(id, self)
    }
}

impl<F> Invoke<ast::PathSegment> for QPathQSelfBuilder<F>
    where F: Invoke<(ast::QSelf, ast::Path)>,
{
    type Result = F::Result;

    fn invoke(mut self, segment: ast::PathSegment) -> F::Result {
        self.path.segments.push(segment);
        self.builder.build(self.qself, self.path)
    }
}
