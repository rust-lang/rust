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
use ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{StableHasher, StableHasherResult,
                                           HashStable};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use syntax::ast;
use ty::{self, Ty, TyCtxt};

use self::SimplifiedTypeGen::*;

pub type SimplifiedType = SimplifiedTypeGen<DefId>;

/// See `simplify_type`
///
/// Note that we keep this type generic over the type of identifier it uses
/// because we sometimes need to use SimplifiedTypeGen values as stable sorting
/// keys (in which case we use a DefPathHash as id-type) but in the general case
/// the non-stable but fast to construct DefId-version is the better choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SimplifiedTypeGen<D>
    where D: Copy + Debug + Ord + Eq + Hash
{
    BoolSimplifiedType,
    CharSimplifiedType,
    IntSimplifiedType(ast::IntTy),
    UintSimplifiedType(ast::UintTy),
    FloatSimplifiedType(ast::FloatTy),
    AdtSimplifiedType(D),
    StrSimplifiedType,
    ArraySimplifiedType,
    PtrSimplifiedType,
    NeverSimplifiedType,
    TupleSimplifiedType(usize),
    TraitSimplifiedType(D),
    ClosureSimplifiedType(D),
    GeneratorSimplifiedType(D),
    AnonSimplifiedType(D),
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
        ty::TyFnDef(def_id, _) |
        ty::TyClosure(def_id, _) => {
            Some(ClosureSimplifiedType(def_id))
        }
        ty::TyGenerator(def_id, _, _) => {
            Some(GeneratorSimplifiedType(def_id))
        }
        ty::TyNever => Some(NeverSimplifiedType),
        ty::TyTuple(ref tys, _) => {
            Some(TupleSimplifiedType(tys.len()))
        }
        ty::TyFnPtr(ref f) => {
            Some(FunctionSimplifiedType(f.skip_binder().inputs().len()))
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

impl<D: Copy + Debug + Ord + Eq + Hash> SimplifiedTypeGen<D> {
    pub fn map_def<U, F>(self, map: F) -> SimplifiedTypeGen<U>
        where F: Fn(D) -> U,
              U: Copy + Debug + Ord + Eq + Hash,
    {
        match self {
            BoolSimplifiedType => BoolSimplifiedType,
            CharSimplifiedType => CharSimplifiedType,
            IntSimplifiedType(t) => IntSimplifiedType(t),
            UintSimplifiedType(t) => UintSimplifiedType(t),
            FloatSimplifiedType(t) => FloatSimplifiedType(t),
            AdtSimplifiedType(d) => AdtSimplifiedType(map(d)),
            StrSimplifiedType => StrSimplifiedType,
            ArraySimplifiedType => ArraySimplifiedType,
            PtrSimplifiedType => PtrSimplifiedType,
            NeverSimplifiedType => NeverSimplifiedType,
            TupleSimplifiedType(n) => TupleSimplifiedType(n),
            TraitSimplifiedType(d) => TraitSimplifiedType(map(d)),
            ClosureSimplifiedType(d) => ClosureSimplifiedType(map(d)),
            GeneratorSimplifiedType(d) => GeneratorSimplifiedType(map(d)),
            AnonSimplifiedType(d) => AnonSimplifiedType(map(d)),
            FunctionSimplifiedType(n) => FunctionSimplifiedType(n),
            ParameterSimplifiedType => ParameterSimplifiedType,
        }
    }
}

impl<'gcx, D> HashStable<StableHashingContext<'gcx>> for SimplifiedTypeGen<D>
    where D: Copy + Debug + Ord + Eq + Hash +
             HashStable<StableHashingContext<'gcx>>,
{
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'gcx>,
                                          hasher: &mut StableHasher<W>) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            BoolSimplifiedType |
            CharSimplifiedType |
            StrSimplifiedType |
            ArraySimplifiedType |
            PtrSimplifiedType |
            NeverSimplifiedType |
            ParameterSimplifiedType => {
                // nothing to do
            }
            IntSimplifiedType(t) => t.hash_stable(hcx, hasher),
            UintSimplifiedType(t) => t.hash_stable(hcx, hasher),
            FloatSimplifiedType(t) => t.hash_stable(hcx, hasher),
            AdtSimplifiedType(d) => d.hash_stable(hcx, hasher),
            TupleSimplifiedType(n) => n.hash_stable(hcx, hasher),
            TraitSimplifiedType(d) => d.hash_stable(hcx, hasher),
            ClosureSimplifiedType(d) => d.hash_stable(hcx, hasher),
            GeneratorSimplifiedType(d) => d.hash_stable(hcx, hasher),
            AnonSimplifiedType(d) => d.hash_stable(hcx, hasher),
            FunctionSimplifiedType(n) => n.hash_stable(hcx, hasher),
        }
    }
}
