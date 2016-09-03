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
pub enum Def {
    Fn(DefId),
    SelfTy(Option<DefId> /* trait */, Option<ast::NodeId> /* impl */),
    Mod(DefId),
    ForeignMod(DefId),
    Static(DefId, bool /* is_mutbl */),
    Const(DefId),
    AssociatedConst(DefId),
    Local(DefId, // def id of variable
             ast::NodeId), // node id of variable
    Variant(DefId /* enum */, DefId /* variant */),
    Enum(DefId),
    TyAlias(DefId),
    AssociatedTy(DefId /* trait */, DefId),
    Trait(DefId),
    PrimTy(hir::PrimTy),
    TyParam(DefId),
    Upvar(DefId,        // def id of closed over local
             ast::NodeId,  // node id of closed over local
             usize,        // index in the freevars list of the closure
             ast::NodeId), // expr node that creates the closure

    // If Def::Struct lives in type namespace it denotes a struct item and its DefId refers
    // to NodeId of the struct itself.
    // If Def::Struct lives in value namespace (e.g. tuple struct, unit struct expressions)
    // it denotes a constructor and its DefId refers to NodeId of the struct's constructor.
    Struct(DefId),
    Union(DefId),
    Label(ast::NodeId),
    Method(DefId),
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

    /// Get the definition, if fully resolved, otherwise panic.
    pub fn full_def(&self) -> Def {
        if self.depth != 0 {
            bug!("path not fully resolved: {:?}", self);
        }
        self.base_def
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

#[derive(Copy, Clone)]
pub struct Export {
    pub name: ast::Name,    // The name of the target.
    pub def_id: DefId, // The definition of the target.
}

impl Def {
    pub fn var_id(&self) -> ast::NodeId {
        match *self {
            Def::Local(_, id) |
            Def::Upvar(_, id, _, _) => {
                id
            }

            Def::Fn(..) | Def::Mod(..) | Def::ForeignMod(..) | Def::Static(..) |
            Def::Variant(..) | Def::Enum(..) | Def::TyAlias(..) | Def::AssociatedTy(..) |
            Def::TyParam(..) | Def::Struct(..) | Def::Union(..) | Def::Trait(..) |
            Def::Method(..) | Def::Const(..) | Def::AssociatedConst(..) |
            Def::PrimTy(..) | Def::Label(..) | Def::SelfTy(..) | Def::Err => {
                bug!("attempted .var_id() on invalid {:?}", self)
            }
        }
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            Def::Fn(id) | Def::Mod(id) | Def::ForeignMod(id) | Def::Static(id, _) |
            Def::Variant(_, id) | Def::Enum(id) | Def::TyAlias(id) | Def::AssociatedTy(_, id) |
            Def::TyParam(id) | Def::Struct(id) | Def::Union(id) | Def::Trait(id) |
            Def::Method(id) | Def::Const(id) | Def::AssociatedConst(id) |
            Def::Local(id, _) | Def::Upvar(id, _, _, _) => {
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
            Def::ForeignMod(..) => "foreign module",
            Def::Static(..) => "static",
            Def::Variant(..) => "variant",
            Def::Enum(..) => "enum",
            Def::TyAlias(..) => "type",
            Def::AssociatedTy(..) => "associated type",
            Def::Struct(..) => "struct",
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
            Def::Err => "unresolved item",
        }
    }
}
