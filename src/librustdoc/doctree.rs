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
    attrs: Vec<ast::Attribute> ,
    where: Span,
    structs: Vec<Struct> ,
    enums: Vec<Enum> ,
    fns: Vec<Function> ,
    mods: Vec<Module> ,
    id: NodeId,
    typedefs: Vec<Typedef> ,
    statics: Vec<Static> ,
    traits: Vec<Trait> ,
    vis: ast::Visibility,
    impls: Vec<Impl> ,
    foreigns: Vec<ast::ForeignMod> ,
    view_items: Vec<ast::ViewItem> ,
    macros: Vec<Macro> ,
    is_crate: bool,
}

impl Module {
    pub fn new(name: Option<Ident>) -> Module {
        Module {
            name       : name,
            id: 0,
            vis: ast::Private,
            where: syntax::codemap::DUMMY_SP,
            attrs      : Vec::new(),
            structs    : Vec::new(),
            enums      : Vec::new(),
            fns        : Vec::new(),
            mods       : Vec::new(),
            typedefs   : Vec::new(),
            statics    : Vec::new(),
            traits     : Vec::new(),
            impls      : Vec::new(),
            view_items : Vec::new(),
            foreigns   : Vec::new(),
            macros     : Vec::new(),
            is_crate   : false,
        }
    }
}

#[deriving(Show, Clone, Encodable, Decodable)]
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
    TraitBound(ast::TraitRef)
}

pub struct Struct {
    vis: ast::Visibility,
    id: NodeId,
    struct_type: StructType,
    name: Ident,
    generics: ast::Generics,
    attrs: Vec<ast::Attribute> ,
    fields: Vec<ast::StructField> ,
    where: Span,
}

pub struct Enum {
    vis: ast::Visibility,
    variants: Vec<Variant> ,
    generics: ast::Generics,
    attrs: Vec<ast::Attribute> ,
    id: NodeId,
    where: Span,
    name: Ident,
}

pub struct Variant {
    name: Ident,
    attrs: Vec<ast::Attribute> ,
    kind: ast::VariantKind,
    id: ast::NodeId,
    vis: ast::Visibility,
    where: Span,
}

pub struct Function {
    decl: ast::FnDecl,
    attrs: Vec<ast::Attribute> ,
    id: NodeId,
    name: Ident,
    vis: ast::Visibility,
    purity: ast::Purity,
    where: Span,
    generics: ast::Generics,
}

pub struct Typedef {
    ty: ast::P<ast::Ty>,
    gen: ast::Generics,
    name: Ident,
    id: ast::NodeId,
    attrs: Vec<ast::Attribute> ,
    where: Span,
    vis: ast::Visibility,
}

pub struct Static {
    type_: ast::P<ast::Ty>,
    mutability: ast::Mutability,
    expr: @ast::Expr,
    name: Ident,
    attrs: Vec<ast::Attribute> ,
    vis: ast::Visibility,
    id: ast::NodeId,
    where: Span,
}

pub struct Trait {
    name: Ident,
    methods: Vec<ast::TraitMethod> , //should be TraitMethod
    generics: ast::Generics,
    parents: Vec<ast::TraitRef> ,
    attrs: Vec<ast::Attribute> ,
    id: ast::NodeId,
    where: Span,
    vis: ast::Visibility,
}

pub struct Impl {
    generics: ast::Generics,
    trait_: Option<ast::TraitRef>,
    for_: ast::P<ast::Ty>,
    methods: Vec<@ast::Method> ,
    attrs: Vec<ast::Attribute> ,
    where: Span,
    vis: ast::Visibility,
    id: ast::NodeId,
}

pub struct Macro {
    name: Ident,
    id: ast::NodeId,
    attrs: Vec<ast::Attribute> ,
    where: Span,
}

pub fn struct_type_from_def(sd: &ast::StructDef) -> StructType {
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
