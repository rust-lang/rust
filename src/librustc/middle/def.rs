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
pub use self::MethodProvenance::*;

use middle::privacy::LastPrivate;
use middle::subst::ParamSpace;
use util::nodemap::NodeMap;
use syntax::ast;
use syntax::ast_util::local_def;

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Def {
    DefFn(ast::DefId, bool /* is_ctor */),
    DefSelfTy(/* trait id */ ast::NodeId),
    DefMod(ast::DefId),
    DefForeignMod(ast::DefId),
    DefStatic(ast::DefId, bool /* is_mutbl */),
    DefConst(ast::DefId),
    DefLocal(ast::NodeId),
    DefVariant(ast::DefId /* enum */, ast::DefId /* variant */, bool /* is_structure */),
    DefTy(ast::DefId, bool /* is_enum */),
    DefAssociatedTy(ast::DefId /* trait */, ast::DefId),
    DefTrait(ast::DefId),
    DefPrimTy(ast::PrimTy),
    DefTyParam(ParamSpace, u32, ast::DefId, ast::Name),
    DefUse(ast::DefId),
    DefUpvar(ast::NodeId,  // id of closed over local
             ast::NodeId), // expr node that creates the closure

    /// Note that if it's a tuple struct's definition, the node id of the ast::DefId
    /// may either refer to the item definition's id or the StructDef.ctor_id.
    ///
    /// The cases that I have encountered so far are (this is not exhaustive):
    /// - If it's a ty_path referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the item definition's id.
    /// - If it's an ExprPath referring to some tuple struct, then DefMap maps
    ///   it to a def whose id is the StructDef.ctor_id.
    DefStruct(ast::DefId),
    DefRegion(ast::NodeId),
    DefLabel(ast::NodeId),
    DefMethod(ast::DefId /* method */, MethodProvenance),
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
    pub fn def_id(&self) -> ast::DefId {
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
pub type DefMap = NodeMap<PathResolution>;
// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap = NodeMap<Vec<Export>>;

#[derive(Copy, Clone)]
pub struct Export {
    pub name: ast::Name,    // The name of the target.
    pub def_id: ast::DefId, // The definition of the target.
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MethodProvenance {
    FromTrait(ast::DefId),
    FromImpl(ast::DefId),
}

impl MethodProvenance {
    pub fn map<F>(self, f: F) -> MethodProvenance where
        F: FnOnce(ast::DefId) -> ast::DefId,
    {
        match self {
            FromTrait(did) => FromTrait(f(did)),
            FromImpl(did) => FromImpl(f(did))
        }
    }
}

impl Def {
    pub fn local_node_id(&self) -> ast::NodeId {
        let def_id = self.def_id();
        assert_eq!(def_id.krate, ast::LOCAL_CRATE);
        def_id.node
    }

    pub fn def_id(&self) -> ast::DefId {
        match *self {
            DefFn(id, _) | DefMod(id) | DefForeignMod(id) | DefStatic(id, _) |
            DefVariant(_, id, _) | DefTy(id, _) | DefAssociatedTy(_, id) |
            DefTyParam(_, _, id, _) | DefUse(id) | DefStruct(id) | DefTrait(id) |
            DefMethod(id, _) | DefConst(id) => {
                id
            }
            DefLocal(id) |
            DefSelfTy(id) |
            DefUpvar(id, _) |
            DefRegion(id) |
            DefLabel(id) => {
                local_def(id)
            }

            DefPrimTy(_) => panic!("attempted .def_id() on DefPrimTy")
        }
    }

    pub fn variant_def_ids(&self) -> Option<(ast::DefId, ast::DefId)> {
        match *self {
            DefVariant(enum_id, var_id, _) => {
                Some((enum_id, var_id))
            }
            _ => None
        }
    }
}
