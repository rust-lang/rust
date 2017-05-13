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
pub use self::StructType::*;
pub use self::TypeBound::*;

use syntax::abi;
use syntax::ast;
use syntax::ast::{Name, NodeId};
use syntax::attr;
use syntax::ptr::P;
use syntax_pos::{self, Span};

use rustc::hir;
use rustc::hir::def_id::CrateNum;

pub struct Module {
    pub name: Option<Name>,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub where_outer: Span,
    pub where_inner: Span,
    pub extern_crates: Vec<ExternCrate>,
    pub imports: Vec<Import>,
    pub structs: Vec<Struct>,
    pub unions: Vec<Union>,
    pub enums: Vec<Enum>,
    pub fns: Vec<Function>,
    pub mods: Vec<Module>,
    pub id: NodeId,
    pub typedefs: Vec<Typedef>,
    pub statics: Vec<Static>,
    pub constants: Vec<Constant>,
    pub traits: Vec<Trait>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub impls: Vec<Impl>,
    pub def_traits: Vec<DefaultImpl>,
    pub foreigns: Vec<hir::ForeignMod>,
    pub macros: Vec<Macro>,
    pub is_crate: bool,
}

impl Module {
    pub fn new(name: Option<Name>) -> Module {
        Module {
            name       : name,
            id: ast::CRATE_NODE_ID,
            vis: hir::Inherited,
            stab: None,
            depr: None,
            where_outer: syntax_pos::DUMMY_SP,
            where_inner: syntax_pos::DUMMY_SP,
            attrs      : hir::HirVec::new(),
            extern_crates: Vec::new(),
            imports    : Vec::new(),
            structs    : Vec::new(),
            unions     : Vec::new(),
            enums      : Vec::new(),
            fns        : Vec::new(),
            mods       : Vec::new(),
            typedefs   : Vec::new(),
            statics    : Vec::new(),
            constants  : Vec::new(),
            traits     : Vec::new(),
            impls      : Vec::new(),
            def_traits : Vec::new(),
            foreigns   : Vec::new(),
            macros     : Vec::new(),
            is_crate   : false,
        }
    }
}

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, Copy)]
pub enum StructType {
    /// A braced struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A unit struct
    Unit,
}

pub enum TypeBound {
    RegionBound,
    TraitBound(hir::TraitRef)
}

pub struct Struct {
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: NodeId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: hir::Generics,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub fields: hir::HirVec<hir::StructField>,
    pub whence: Span,
}

pub struct Union {
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: NodeId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: hir::Generics,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub fields: hir::HirVec<hir::StructField>,
    pub whence: Span,
}

pub struct Enum {
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub variants: hir::HirVec<Variant>,
    pub generics: hir::Generics,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub id: NodeId,
    pub whence: Span,
    pub name: Name,
}

pub struct Variant {
    pub name: Name,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub def: hir::VariantData,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub whence: Span,
}

pub struct Function {
    pub decl: hir::FnDecl,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub id: NodeId,
    pub name: Name,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub whence: Span,
    pub generics: hir::Generics,
    pub abi: abi::Abi,
    pub body: hir::BodyId,
}

pub struct Typedef {
    pub ty: P<hir::Ty>,
    pub gen: hir::Generics,
    pub name: Name,
    pub id: ast::NodeId,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

#[derive(Debug)]
pub struct Static {
    pub type_: P<hir::Ty>,
    pub mutability: hir::Mutability,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: ast::NodeId,
    pub whence: Span,
}

pub struct Constant {
    pub type_: P<hir::Ty>,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: ast::NodeId,
    pub whence: Span,
}

pub struct Trait {
    pub unsafety: hir::Unsafety,
    pub name: Name,
    pub items: hir::HirVec<hir::TraitItem>,
    pub generics: hir::Generics,
    pub bounds: hir::HirVec<hir::TyParamBound>,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub id: ast::NodeId,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub struct Impl {
    pub unsafety: hir::Unsafety,
    pub polarity: hir::ImplPolarity,
    pub generics: hir::Generics,
    pub trait_: Option<hir::TraitRef>,
    pub for_: P<hir::Ty>,
    pub items: hir::HirVec<hir::ImplItem>,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: ast::NodeId,
}

pub struct DefaultImpl {
    pub unsafety: hir::Unsafety,
    pub trait_: hir::TraitRef,
    pub id: ast::NodeId,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub whence: Span,
}

// For Macro we store the DefId instead of the NodeId, since we also create
// these imported macro_rules (which only have a DUMMY_NODE_ID).
pub struct Macro {
    pub name: Name,
    pub def_id: hir::def_id::DefId,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub matchers: hir::HirVec<Span>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub imported_from: Option<Name>,
}

pub struct ExternCrate {
    pub name: Name,
    pub cnum: CrateNum,
    pub path: Option<String>,
    pub vis: hir::Visibility,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub whence: Span,
}

pub struct Import {
    pub name: Name,
    pub id: NodeId,
    pub vis: hir::Visibility,
    pub attrs: hir::HirVec<ast::Attribute>,
    pub path: hir::Path,
    pub glob: bool,
    pub whence: Span,
}

pub fn struct_type_from_def(vdata: &hir::VariantData) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
