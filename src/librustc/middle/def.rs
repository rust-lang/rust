// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::ast_util::local_def;

use std::gc::Gc;

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum Def {
    DefFn(ast::DefId, ast::FnStyle),
    DefStaticMethod(/* method */ ast::DefId, MethodProvenance, ast::FnStyle),
    DefSelfTy(/* trait id */ ast::NodeId),
    DefMod(ast::DefId),
    DefForeignMod(ast::DefId),
    DefStatic(ast::DefId, bool /* is_mutbl */),
    DefArg(ast::NodeId, ast::BindingMode),
    DefLocal(ast::NodeId, ast::BindingMode),
    DefVariant(ast::DefId /* enum */, ast::DefId /* variant */, bool /* is_structure */),
    DefTy(ast::DefId),
    DefTrait(ast::DefId),
    DefPrimTy(ast::PrimTy),
    DefTyParam(ast::DefId, uint),
    DefBinding(ast::NodeId, ast::BindingMode),
    DefUse(ast::DefId),
    DefUpvar(ast::NodeId,  // id of closed over var
             Gc<Def>,     // closed over def
             ast::NodeId,  // expr node that creates the closure
             ast::NodeId), // id for the block/body of the closure expr

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
    DefMethod(ast::DefId /* method */, Option<ast::DefId> /* trait */),
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub enum MethodProvenance {
    FromTrait(ast::DefId),
    FromImpl(ast::DefId),
}

impl Def {
    pub fn def_id(&self) -> ast::DefId {
        match *self {
            DefFn(id, _) | DefStaticMethod(id, _, _) | DefMod(id) |
            DefForeignMod(id) | DefStatic(id, _) |
            DefVariant(_, id, _) | DefTy(id) | DefTyParam(id, _) |
            DefUse(id) | DefStruct(id) | DefTrait(id) | DefMethod(id, _) => {
                id
            }
            DefArg(id, _) |
            DefLocal(id, _) |
            DefSelfTy(id) |
            DefUpvar(id, _, _, _) |
            DefBinding(id, _) |
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
