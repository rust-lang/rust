// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty::{mod, Ty};
use syntax::ast;

use self::SimplifiedType::*;

/// See `simplify_type
#[deriving(Clone, PartialEq, Eq, Hash)]
pub enum SimplifiedType {
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ast::IntTy),
    UintSimplifiedType(ast::UintTy),
    FloatSimplifiedType(ast::FloatTy),
    EnumSimplifiedType(ast::DefId),
    StrSimplifiedType,
    VecSimplifiedType,
    PtrSimplifiedType,
    TupleSimplifiedType(uint),
    TraitSimplifiedType(ast::DefId),
    StructSimplifiedType(ast::DefId),
    UnboxedClosureSimplifiedType(ast::DefId),
    FunctionSimplifiedType(uint),
    ParameterSimplifiedType,
}

/// Tries to simplify a type by dropping type parameters, deref'ing away any reference types, etc.
/// The idea is to get something simple that we can use to quickly decide if two types could unify
/// during method lookup.
///
/// If `can_simplify_params` is false, then we will fail to simplify type parameters entirely. This
/// is useful when those type parameters would be instantiated with fresh type variables, since
/// then we can't say much about whether two types would unify. Put another way,
/// `can_simplify_params` should be true if type parameters appear free in `ty` and `false` if they
/// are to be considered bound.
pub fn simplify_type(tcx: &ty::ctxt,
                     ty: Ty,
                     can_simplify_params: bool)
                     -> Option<SimplifiedType>
{
    match ty.sty {
        ty::ty_bool => Some(BoolSimplifiedType),
        ty::ty_char => Some(CharSimplifiedType),
        ty::ty_int(int_type) => Some(IntSimplifiedType(int_type)),
        ty::ty_uint(uint_type) => Some(UintSimplifiedType(uint_type)),
        ty::ty_float(float_type) => Some(FloatSimplifiedType(float_type)),
        ty::ty_enum(def_id, _) => Some(EnumSimplifiedType(def_id)),
        ty::ty_str => Some(StrSimplifiedType),
        ty::ty_vec(..) => Some(VecSimplifiedType),
        ty::ty_ptr(_) => Some(PtrSimplifiedType),
        ty::ty_trait(ref trait_info) => {
            Some(TraitSimplifiedType(trait_info.principal.def_id))
        }
        ty::ty_struct(def_id, _) => {
            Some(StructSimplifiedType(def_id))
        }
        ty::ty_rptr(_, mt) => {
            // since we introduce auto-refs during method lookup, we
            // just treat &T and T as equivalent from the point of
            // view of possibly unifying
            simplify_type(tcx, mt.ty, can_simplify_params)
        }
        ty::ty_uniq(_) => {
            // treat like we would treat `Box`
            let def_id = tcx.lang_items.owned_box().unwrap();
            Some(StructSimplifiedType(def_id))
        }
        ty::ty_unboxed_closure(def_id, _, _) => {
            Some(UnboxedClosureSimplifiedType(def_id))
        }
        ty::ty_tup(ref tys) => {
            Some(TupleSimplifiedType(tys.len()))
        }
        ty::ty_closure(ref f) => {
            Some(FunctionSimplifiedType(f.sig.inputs.len()))
        }
        ty::ty_bare_fn(ref f) => {
            Some(FunctionSimplifiedType(f.sig.inputs.len()))
        }
        ty::ty_param(_) => {
            if can_simplify_params {
                Some(ParameterSimplifiedType)
            } else {
                None
            }
        }
        ty::ty_open(_) | ty::ty_infer(_) | ty::ty_err => None,
    }
}

