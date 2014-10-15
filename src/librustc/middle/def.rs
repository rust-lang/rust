// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::subst::ParamSpace;
use syntax::ast;
use syntax::ast_util::local_def;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash, Show)]
pub enum Def {
    DefFn(ast::DefId, ast::FnStyle, bool /* is_ctor */),
    DefStaticMethod(/* method */ ast::DefId, MethodProvenance, ast::FnStyle),
    DefSelfTy(/* trait id */ ast::NodeId),
    DefMod(ast::DefId),
    DefForeignMod(ast::DefId),
    DefStatic(ast::DefId, bool /* is_mutbl */),
    DefConst(ast::DefId),
    DefLocal(ast::NodeId),
    DefVariant(ast::DefId /* enum */, ast::DefId /* variant */, bool /* is_structure */),
    DefTy(ast::DefId, bool /* is_enum */),
    DefAssociatedTy(ast::DefId),
    DefTrait(ast::DefId),
    DefPrimTy(ast::PrimTy),
    DefTyParam(ParamSpace, ast::DefId, uint),
    DefUse(ast::DefId),
    DefUpvar(ast::NodeId,  // id of closed over local
             ast::NodeId,  // expr node that creates the closure
             ast::NodeId), // block node for the closest enclosing proc
                           // or unboxed closure, DUMMY_NODE_ID otherwise

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

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash, Show)]
pub enum MethodProvenance {
    FromTrait(ast::DefId),
    FromImpl(ast::DefId),
}

impl Def {
    pub fn def_id(&self) -> ast::DefId {
        match *self {
            DefFn(id, _, _) | DefStaticMethod(id, _, _) | DefMod(id) |
            DefForeignMod(id) | DefStatic(id, _) |
            DefVariant(_, id, _) | DefTy(id, _) | DefAssociatedTy(id) |
            DefTyParam(_, id, _) | DefUse(id) | DefStruct(id) | DefTrait(id) |
            DefMethod(id, _, _) | DefConst(id) => {
                id
            }
            DefLocal(id) |
            DefSelfTy(id) |
            DefUpvar(id, _, _) |
            DefRegion(id) |
            DefTyParamBinder(id) |
            DefLabel(id) => {
                local_def(id)
            }

            DefPrimTy(_) => fail!()
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

