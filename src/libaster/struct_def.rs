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

use attr::AttrBuilder;
use ident::ToIdent;
use invoke::{Invoke, Identity};
use ty::TyBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct StructDefBuilder<F=Identity> {
    callback: F,
    span: Span,
    fields: Vec<ast::StructField>,
}

impl StructDefBuilder {
    pub fn new() -> Self {
        StructDefBuilder::new_with_callback(Identity)
    }
}

impl<F> StructDefBuilder<F>
    where F: Invoke<P<ast::StructDef>>
{
    pub fn new_with_callback(callback: F) -> Self {
        StructDefBuilder {
            callback: callback,
            span: DUMMY_SP,
            fields: vec![],
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn with_fields<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=ast::StructField>,
    {
        self.fields.extend(iter);
        self
    }

    pub fn with_field(mut self, field: ast::StructField) -> Self {
        self.fields.push(field);
        self
    }

    pub fn field<T>(self, id: T) -> StructFieldBuilder<Self>
        where T: ToIdent,
    {
        let span = self.span;
        StructFieldBuilder::named_with_callback(id, self).span(span)
    }

    pub fn build(self) -> F::Result {
        self.callback.invoke(P(ast::StructDef {
            fields: self.fields,
            ctor_id: None,
        }))
    }
}

impl<F> Invoke<ast::StructField> for StructDefBuilder<F>
    where F: Invoke<P<ast::StructDef>>,
{
    type Result = Self;

    fn invoke(self, field: ast::StructField) -> Self {
        self.with_field(field)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct StructFieldBuilder<F=Identity> {
    callback: F,
    span: Span,
    kind: ast::StructFieldKind,
    attrs: Vec<ast::Attribute>,
}

impl StructFieldBuilder {
    pub fn named<T>(name: T) -> Self
        where T: ToIdent,
    {
        StructFieldBuilder::named_with_callback(name, Identity)
    }

    pub fn unnamed() -> Self {
        StructFieldBuilder::unnamed_with_callback(Identity)
    }
}

impl<F> StructFieldBuilder<F>
    where F: Invoke<ast::StructField>,
{
    pub fn named_with_callback<T>(id: T, callback: F) -> Self
        where T: ToIdent,
    {
        let id = id.to_ident();
        StructFieldBuilder {
            callback: callback,
            span: DUMMY_SP,
            kind: ast::StructFieldKind::NamedField(id, ast::Inherited),
            attrs: vec![],
        }
    }

    pub fn unnamed_with_callback(callback: F) -> Self {
        StructFieldBuilder {
            callback: callback,
            span: DUMMY_SP,
            kind: ast::StructFieldKind::UnnamedField(ast::Inherited),
            attrs: vec![],
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn pub_(mut self) -> Self {
        match self.kind {
            ast::StructFieldKind::NamedField(_, ref mut vis) => { *vis = ast::Public; }
            ast::StructFieldKind::UnnamedField(ref mut vis) => { *vis = ast::Public; }
        }
        self
    }

    pub fn attr(self) -> AttrBuilder<Self> {
        let span = self.span;
        AttrBuilder::new_with_callback(self).span(span)
    }

    pub fn build_ty(self, ty: P<ast::Ty>) -> F::Result {
        let field = ast::StructField_ {
            kind: self.kind,
            id: ast::DUMMY_NODE_ID,
            ty: ty,
            attrs: self.attrs,
        };
        self.callback.invoke(respan(self.span, field))
    }

    pub fn ty(self) -> TyBuilder<Self> {
        let span = self.span;
        TyBuilder::new_with_callback(self).span(span)
    }
}

impl<F> Invoke<ast::Attribute> for StructFieldBuilder<F> {
    type Result = Self;

    fn invoke(mut self, attr: ast::Attribute) -> Self {
        self.attrs.push(attr);
        self
    }
}

impl<F> Invoke<P<ast::Ty>> for StructFieldBuilder<F>
    where F: Invoke<ast::StructField>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.build_ty(ty)
    }
}
