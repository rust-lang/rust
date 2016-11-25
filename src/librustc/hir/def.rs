// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::def_id::DefId;
use util::nodemap::NodeMap;
use syntax::ast;
use hir;

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum CtorKind {
    // Constructor function automatically created by a tuple struct/variant.
    Fn,
    // Constructor constant automatically created by a unit struct/variant.
    Const,
    // Unusable name in value namespace created by a struct variant.
    Fictive,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Def {
    // Type namespace
    Mod(DefId),
    Struct(DefId), // DefId refers to NodeId of the struct itself
    Union(DefId),
    Enum(DefId),
    Variant(DefId),
    Trait(DefId),
    TyAlias(DefId),
    AssociatedTy(DefId),
    PrimTy(hir::PrimTy),
    TyParam(DefId),
    SelfTy(Option<DefId> /* trait */, Option<DefId> /* impl */),

    // Value namespace
    Fn(DefId),
    Const(DefId),
    Static(DefId, bool /* is_mutbl */),
    StructCtor(DefId, CtorKind), // DefId refers to NodeId of the struct's constructor
    VariantCtor(DefId, CtorKind),
    Method(DefId),
    AssociatedConst(DefId),
    Local(DefId),
    Upvar(DefId,        // def id of closed over local
          usize,        // index in the freevars list of the closure
          ast::NodeId), // expr node that creates the closure
    Label(ast::NodeId),

    // Macro namespace
    Macro(DefId),

    // Both namespaces
    Err,
}

/// The result of resolving a path.
/// Before type checking completes, `depth` represents the number of
/// trailing segments which are yet unresolved. Afterwards, if there
/// were no errors, all paths should be fully resolved, with `depth`
/// set to `0` and `base_def` representing the final resolution.
///
///     module::Type::AssocX::AssocY::MethodOrAssocType
///     ^~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///     base_def      depth = 3
///
///     <T as Trait>::AssocX::AssocY::MethodOrAssocType
///           ^~~~~~~~~~~~~~  ^~~~~~~~~~~~~~~~~~~~~~~~~
///           base_def        depth = 2
#[derive(Copy, Clone, Debug)]
pub struct PathResolution {
    pub base_def: Def,
    pub depth: usize
}

impl PathResolution {
    pub fn new(def: Def) -> PathResolution {
        PathResolution { base_def: def, depth: 0 }
    }

    pub fn kind_name(&self) -> &'static str {
        if self.depth != 0 {
            "associated item"
        } else {
            self.base_def.kind_name()
        }
    }
}

// Definition mapping
pub type DefMap = NodeMap<PathResolution>;
// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap = NodeMap<Vec<Export>>;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct Export {
    pub name: ast::Name, // The name of the target.
    pub def: Def, // The definition of the target.
}

impl CtorKind {
    pub fn from_ast(vdata: &ast::VariantData) -> CtorKind {
        match *vdata {
            ast::VariantData::Tuple(..) => CtorKind::Fn,
            ast::VariantData::Unit(..) => CtorKind::Const,
            ast::VariantData::Struct(..) => CtorKind::Fictive,
        }
    }
    pub fn from_hir(vdata: &hir::VariantData) -> CtorKind {
        match *vdata {
            hir::VariantData::Tuple(..) => CtorKind::Fn,
            hir::VariantData::Unit(..) => CtorKind::Const,
            hir::VariantData::Struct(..) => CtorKind::Fictive,
        }
    }
}

impl Def {
    pub fn def_id(&self) -> DefId {
        match *self {
            Def::Fn(id) | Def::Mod(id) | Def::Static(id, _) |
            Def::Variant(id) | Def::VariantCtor(id, ..) | Def::Enum(id) | Def::TyAlias(id) |
            Def::AssociatedTy(id) | Def::TyParam(id) | Def::Struct(id) | Def::StructCtor(id, ..) |
            Def::Union(id) | Def::Trait(id) | Def::Method(id) | Def::Const(id) |
            Def::AssociatedConst(id) | Def::Local(id) | Def::Upvar(id, ..) | Def::Macro(id) => {
                id
            }

            Def::Label(..)  |
            Def::PrimTy(..) |
            Def::SelfTy(..) |
            Def::Err => {
                bug!("attempted .def_id() on invalid def: {:?}", self)
            }
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match *self {
            Def::Fn(..) => "function",
            Def::Mod(..) => "module",
            Def::Static(..) => "static",
            Def::Variant(..) => "variant",
            Def::VariantCtor(.., CtorKind::Fn) => "tuple variant",
            Def::VariantCtor(.., CtorKind::Const) => "unit variant",
            Def::VariantCtor(.., CtorKind::Fictive) => "struct variant",
            Def::Enum(..) => "enum",
            Def::TyAlias(..) => "type alias",
            Def::AssociatedTy(..) => "associated type",
            Def::Struct(..) => "struct",
            Def::StructCtor(.., CtorKind::Fn) => "tuple struct",
            Def::StructCtor(.., CtorKind::Const) => "unit struct",
            Def::StructCtor(.., CtorKind::Fictive) => bug!("impossible struct constructor"),
            Def::Union(..) => "union",
            Def::Trait(..) => "trait",
            Def::Method(..) => "method",
            Def::Const(..) => "constant",
            Def::AssociatedConst(..) => "associated constant",
            Def::TyParam(..) => "type parameter",
            Def::PrimTy(..) => "builtin type",
            Def::Local(..) => "local variable",
            Def::Upvar(..) => "closure capture",
            Def::Label(..) => "label",
            Def::SelfTy(..) => "self type",
            Def::Macro(..) => "macro",
            Def::Err => "unresolved item",
        }
    }
}
