// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Def::*;

use middle::def_id::DefId;
use middle::privacy::LastPrivate;
use middle::subst::ParamSpace;
use util::nodemap::NodeMap;
use syntax::ast;
use rustc_front::hir;

use std::cell::RefCell;

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Def {
    DefFn(DefId, bool /* is_ctor */),
    DefSelfTy(Option<DefId>,                    // trait id
              Option<(ast::NodeId, ast::NodeId)>),   // (impl id, self type id)
    DefMod(DefId),
    DefForeignMod(DefId),
    DefStatic(DefId, bool /* is_mutbl */),
    DefConst(DefId),
    DefAssociatedConst(DefId),
    DefLocal(DefId, // def id of variable
             ast::NodeId), // node id of variable
    DefVariant(DefId /* enum */, DefId /* variant */, bool /* is_structure */),
    DefTy(DefId, bool /* is_enum */),
    DefAssociatedTy(DefId /* trait */, DefId),
    DefTrait(DefId),
    DefPrimTy(hir::PrimTy),
    DefTyParam(ParamSpace, u32, DefId, ast::Name),
    DefUse(DefId),
    DefUpvar(DefId,        // def id of closed over local
             ast::NodeId,  // node id of closed over local
             usize,        // index in the freevars list of the closure
             ast::NodeId), // expr node that creates the closure

    /// Note that if it's a tuple struct's definition, the node id of the DefId
    /// may either refer to the item definition's id or the VariantData.ctor_id.
    ///
    /// The cases that I have encountered so far are (this is not exhaustive):
    /// - If it's a ty_path referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the item definition's id.
    /// - If it's an ExprPath referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the VariantData.ctor_id.
    DefStruct(DefId),
    DefLabel(ast::NodeId),
    DefMethod(DefId),
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
    pub last_private: LastPrivate,
    pub depth: usize
}

impl PathResolution {
    /// Get the definition, if fully resolved, otherwise panic.
    pub fn full_def(&self) -> Def {
        if self.depth != 0 {
            panic!("path not fully resolved: {:?}", self);
        }
        self.base_def
    }

    /// Get the DefId, if fully resolved, otherwise panic.
    pub fn def_id(&self) -> DefId {
        self.full_def().def_id()
    }

    pub fn new(base_def: Def,
               last_private: LastPrivate,
               depth: usize)
               -> PathResolution {
        PathResolution {
            base_def: base_def,
            last_private: last_private,
            depth: depth,
        }
    }
}

// Definition mapping
pub type DefMap = RefCell<NodeMap<PathResolution>>;
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
            DefLocal(_, id) |
            DefUpvar(_, id, _, _) => {
                id
            }

            DefFn(..) | DefMod(..) | DefForeignMod(..) | DefStatic(..) |
            DefVariant(..) | DefTy(..) | DefAssociatedTy(..) |
            DefTyParam(..) | DefUse(..) | DefStruct(..) | DefTrait(..) |
            DefMethod(..) | DefConst(..) | DefAssociatedConst(..) |
            DefPrimTy(..) | DefLabel(..) | DefSelfTy(..) => {
                panic!("attempted .def_id() on invalid {:?}", self)
            }
        }
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            DefFn(id, _) | DefMod(id) | DefForeignMod(id) | DefStatic(id, _) |
            DefVariant(_, id, _) | DefTy(id, _) | DefAssociatedTy(_, id) |
            DefTyParam(_, _, id, _) | DefUse(id) | DefStruct(id) | DefTrait(id) |
            DefMethod(id) | DefConst(id) | DefAssociatedConst(id) |
            DefLocal(id, _) | DefUpvar(id, _, _, _) => {
                id
            }

            DefLabel(..)  |
            DefPrimTy(..) |
            DefSelfTy(..) => {
                panic!("attempted .def_id() on invalid def: {:?}", self)
            }
        }
    }

    pub fn variant_def_ids(&self) -> Option<(DefId, DefId)> {
        match *self {
            DefVariant(enum_id, var_id, _) => {
                Some((enum_id, var_id))
            }
            _ => None
        }
    }
}
