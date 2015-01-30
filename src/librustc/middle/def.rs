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
pub use self::TraitItemKind::*;

use middle::subst::ParamSpace;
use middle::ty::{ExplicitSelfCategory, StaticExplicitSelfCategory};
use util::nodemap::NodeMap;
use syntax::ast;
use syntax::ast_util::local_def;

use std::cell::RefCell;

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum Def {
    DefFn(ast::DefId, bool /* is_ctor */),
    DefStaticMethod(/* method */ ast::DefId, MethodProvenance),
    DefSelfTy(/* trait id */ ast::NodeId),
    DefMod(ast::DefId),
    DefForeignMod(ast::DefId),
    DefStatic(ast::DefId, bool /* is_mutbl */),
    DefConst(ast::DefId),
    DefLocal(ast::NodeId),
    DefVariant(ast::DefId /* enum */, ast::DefId /* variant */, bool /* is_structure */),
    DefTy(ast::DefId, bool /* is_enum */),
    DefAssociatedTy(ast::DefId /* trait */, ast::DefId),
    // A partially resolved path to an associated type `T::U` where `T` is a concrete
    // type (indicated by the DefId) which implements a trait which has an associated
    // type `U` (indicated by the Ident).
    // FIXME(#20301) -- should use Name
    DefAssociatedPath(TyParamProvenance, ast::Ident),
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
    DefTyParamBinder(ast::NodeId), /* struct, impl or trait with ty params */
    DefRegion(ast::NodeId),
    DefLabel(ast::NodeId),
    DefMethod(ast::DefId /* method */, Option<ast::DefId> /* trait */, MethodProvenance),
}

// Definition mapping
pub type DefMap = RefCell<NodeMap<Def>>;
// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap = NodeMap<Vec<Export>>;

#[derive(Copy)]
pub struct Export {
    pub name: ast::Name,    // The name of the target.
    pub def_id: ast::DefId, // The definition of the target.
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum MethodProvenance {
    FromTrait(ast::DefId),
    FromImpl(ast::DefId),
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum TyParamProvenance {
    FromSelf(ast::DefId),
    FromParam(ast::DefId),
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

impl TyParamProvenance {
    pub fn def_id(&self) -> ast::DefId {
        match *self {
            TyParamProvenance::FromSelf(ref did) => did.clone(),
            TyParamProvenance::FromParam(ref did) => did.clone(),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TraitItemKind {
    NonstaticMethodTraitItemKind,
    StaticMethodTraitItemKind,
    TypeTraitItemKind,
}

impl TraitItemKind {
    pub fn from_explicit_self_category(explicit_self_category:
                                       ExplicitSelfCategory)
                                       -> TraitItemKind {
        if explicit_self_category == StaticExplicitSelfCategory {
            StaticMethodTraitItemKind
        } else {
            NonstaticMethodTraitItemKind
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
            DefFn(id, _) | DefStaticMethod(id, _) | DefMod(id) |
            DefForeignMod(id) | DefStatic(id, _) |
            DefVariant(_, id, _) | DefTy(id, _) | DefAssociatedTy(_, id) |
            DefTyParam(_, _, id, _) | DefUse(id) | DefStruct(id) | DefTrait(id) |
            DefMethod(id, _, _) | DefConst(id) |
            DefAssociatedPath(TyParamProvenance::FromSelf(id), _) |
            DefAssociatedPath(TyParamProvenance::FromParam(id), _) => {
                id
            }
            DefLocal(id) |
            DefSelfTy(id) |
            DefUpvar(id, _) |
            DefRegion(id) |
            DefTyParamBinder(id) |
            DefLabel(id) => {
                local_def(id)
            }

            DefPrimTy(_) => panic!()
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
