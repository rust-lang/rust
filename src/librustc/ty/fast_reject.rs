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
use ty::{self, Ty, TyCtxt};
use syntax::ast;
use middle::lang_items::OwnedBoxLangItem;

use self::SimplifiedType::*;

/// See `simplify_type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SimplifiedType {
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ast::IntTy),
    UintSimplifiedType(ast::UintTy),
    FloatSimplifiedType(ast::FloatTy),
    AdtSimplifiedType(DefId),
    StrSimplifiedType,
    ArraySimplifiedType,
    PtrSimplifiedType,
    NeverSimplifiedType,
    TupleSimplifiedType(usize),
    TraitSimplifiedType(DefId),
    ClosureSimplifiedType(DefId),
    AnonSimplifiedType(DefId),
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
pub fn simplify_type<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
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
        ty::TyAdt(def, _) => Some(AdtSimplifiedType(def.did)),
        ty::TyStr => Some(StrSimplifiedType),
        ty::TyArray(..) | ty::TySlice(_) => Some(ArraySimplifiedType),
        ty::TyRawPtr(_) => Some(PtrSimplifiedType),
        ty::TyDynamic(ref trait_info, ..) => {
            trait_info.principal().map(|p| TraitSimplifiedType(p.def_id()))
        }
        ty::TyRef(_, mt) => {
            // since we introduce auto-refs during method lookup, we
            // just treat &T and T as equivalent from the point of
            // view of possibly unifying
            simplify_type(tcx, mt.ty, can_simplify_params)
        }
        ty::TyBox(_) => {
            // treat like we would treat `Box`
            Some(AdtSimplifiedType(tcx.require_lang_item(OwnedBoxLangItem)))
        }
        ty::TyClosure(def_id, _) => {
            Some(ClosureSimplifiedType(def_id))
        }
        ty::TyNever => Some(NeverSimplifiedType),
        ty::TyTuple(ref tys) => {
            Some(TupleSimplifiedType(tys.len()))
        }
        ty::TyFnDef(.., ref f) | ty::TyFnPtr(ref f) => {
            Some(FunctionSimplifiedType(f.sig.skip_binder().inputs().len()))
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
        ty::TyAnon(def_id, _) => {
            Some(AnonSimplifiedType(def_id))
        }
        ty::TyInfer(_) | ty::TyError => None,
    }
}
