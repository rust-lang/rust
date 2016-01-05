// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use middle::ty::{self, Ty};
use syntax::ast;

use self::SimplifiedType::*;

/// See `simplify_type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimplifiedType {
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ast::IntTy),
    UintSimplifiedType(ast::UintTy),
    FloatSimplifiedType(ast::FloatTy),
    EnumSimplifiedType(DefId),
    StrSimplifiedType,
    VecSimplifiedType,
    PtrSimplifiedType,
    TupleSimplifiedType(usize),
    TraitSimplifiedType(DefId),
    StructSimplifiedType(DefId),
    ClosureSimplifiedType(DefId),
    FunctionSimplifiedType(usize),
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
        ty::TyBool => Some(BoolSimplifiedType),
        ty::TyChar => Some(CharSimplifiedType),
        ty::TyInt(int_type) => Some(IntSimplifiedType(int_type)),
        ty::TyUint(uint_type) => Some(UintSimplifiedType(uint_type)),
        ty::TyFloat(float_type) => Some(FloatSimplifiedType(float_type)),
        ty::TyEnum(def, _) => Some(EnumSimplifiedType(def.did)),
        ty::TyStr => Some(StrSimplifiedType),
        ty::TyArray(..) | ty::TySlice(_) => Some(VecSimplifiedType),
        ty::TyRawPtr(_) => Some(PtrSimplifiedType),
        ty::TyTrait(ref trait_info) => {
            Some(TraitSimplifiedType(trait_info.principal_def_id()))
        }
        ty::TyStruct(def, _) => {
            Some(StructSimplifiedType(def.did))
        }
        ty::TyRef(_, mt) => {
            // since we introduce auto-refs during method lookup, we
            // just treat &T and T as equivalent from the point of
            // view of possibly unifying
            simplify_type(tcx, mt.ty, can_simplify_params)
        }
        ty::TyBox(_) => {
            // treat like we would treat `Box`
            match tcx.lang_items.require_owned_box() {
                Ok(def_id) => Some(StructSimplifiedType(def_id)),
                Err(msg) => tcx.sess.fatal(&msg),
            }
        }
        ty::TyClosure(def_id, _) => {
            Some(ClosureSimplifiedType(def_id))
        }
        ty::TyTuple(ref tys) => {
            Some(TupleSimplifiedType(tys.len()))
        }
        ty::TyBareFn(_, ref f) => {
            Some(FunctionSimplifiedType(f.sig.0.inputs.len()))
        }
        ty::TyProjection(_) | ty::TyParam(_) => {
            if can_simplify_params {
                // In normalized types, projections don't unify with
                // anything. when lazy normalization happens, this
                // will change. It would still be nice to have a way
                // to deal with known-not-to-unify-with-anything
                // projections (e.g. the likes of <__S as Encoder>::Error).
                Some(ParameterSimplifiedType)
            } else {
                None
            }
        }
        ty::TyInfer(_) | ty::TyError => None,
    }
}
