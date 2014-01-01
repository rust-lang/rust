// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.

use syntax;
use syntax::codemap::Span;
use syntax::ast;
use syntax::ast::{Ident, NodeId};

pub struct Module {
    name: Option<Ident>,
    attrs: ~[ast::Attribute],
    where: Span,
    structs: ~[Struct],
    enums: ~[Enum],
    fns: ~[Function],
    mods: ~[Module],
    id: NodeId,
    typedefs: ~[Typedef],
    statics: ~[Static],
    traits: ~[Trait],
    vis: ast::visibility,
    impls: ~[Impl],
    foreigns: ~[ast::foreign_mod],
    view_items: ~[ast::view_item],
}

impl Module {
    pub fn new(name: Option<Ident>) -> Module {
        Module {
            name       : name,
            id: 0,
            vis: ast::private,
            where: syntax::codemap::DUMMY_SP,
            attrs      : ~[],
            structs    : ~[],
            enums      : ~[],
            fns        : ~[],
            mods       : ~[],
            typedefs   : ~[],
            statics    : ~[],
            traits     : ~[],
            impls      : ~[],
            view_items : ~[],
            foreigns   : ~[],
        }
    }
}

#[deriving(ToStr, Clone, Encodable, Decodable)]
pub enum StructType {
    /// A normal struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A newtype struct (tuple struct with one element)
    Newtype,
    /// A unit struct
    Unit
}

pub enum TypeBound {
    RegionBound,
    TraitBound(ast::trait_ref)
}

pub struct Struct {
    vis: ast::visibility,
    id: NodeId,
    struct_type: StructType,
    name: Ident,
    generics: ast::Generics,
    attrs: ~[ast::Attribute],
    fields: ~[ast::struct_field],
    where: Span,
}

pub struct Enum {
    vis: ast::visibility,
    variants: ~[Variant],
    generics: ast::Generics,
    attrs: ~[ast::Attribute],
    id: NodeId,
    where: Span,
    name: Ident,
}

pub struct Variant {
    name: Ident,
    attrs: ~[ast::Attribute],
    kind: ast::variant_kind,
    id: ast::NodeId,
    vis: ast::visibility,
    where: Span,
}

pub struct Function {
    decl: ast::fn_decl,
    attrs: ~[ast::Attribute],
    id: NodeId,
    name: Ident,
    vis: ast::visibility,
    purity: ast::purity,
    where: Span,
    generics: ast::Generics,
}

pub struct Typedef {
    ty: ast::P<ast::Ty>,
    gen: ast::Generics,
    name: Ident,
    id: ast::NodeId,
    attrs: ~[ast::Attribute],
    where: Span,
    vis: ast::visibility,
}

pub struct Static {
    type_: ast::P<ast::Ty>,
    mutability: ast::Mutability,
    expr: @ast::Expr,
    name: Ident,
    attrs: ~[ast::Attribute],
    vis: ast::visibility,
    id: ast::NodeId,
    where: Span,
}

pub struct Trait {
    name: Ident,
    methods: ~[ast::trait_method], //should be TraitMethod
    generics: ast::Generics,
    parents: ~[ast::trait_ref],
    attrs: ~[ast::Attribute],
    id: ast::NodeId,
    where: Span,
    vis: ast::visibility,
}

pub struct Impl {
    generics: ast::Generics,
    trait_: Option<ast::trait_ref>,
    for_: ast::P<ast::Ty>,
    methods: ~[@ast::method],
    attrs: ~[ast::Attribute],
    where: Span,
    vis: ast::visibility,
    id: ast::NodeId,
}

pub fn struct_type_from_def(sd: &ast::struct_def) -> StructType {
    if sd.ctor_id.is_some() {
        // We are in a tuple-struct
        match sd.fields.len() {
            0 => Unit,
            1 => Newtype,
            _ => Tuple
        }
    } else {
        Plain
    }
}
