//! Freshening is the process of replacing unknown variables with fresh types. The idea is that
//! the type, after freshening, contains no inference variables but instead contains either a
//! value for each variable or fresh "arbitrary" types wherever a variable would have been.
//!
//! Freshening is used primarily to get a good type for inserting into a cache. The result
//! summarizes what the type inferencer knows "so far". The primary place it is used right now is
//! in the trait matching algorithm, which needs to be able to cache whether an `impl` self type
//! matches some other type X -- *without* affecting `X`. That means that if the type `X` is in
//! fact an unbound type variable, we want the match to be regarded as ambiguous, because depending
//! on what type that type variable is ultimately assigned, the match may or may not succeed.
//!
//! To handle closures, freshened types also have to contain the signature and kind of any
//! closure in the local inference context, as otherwise the cache key might be invalidated.
//! The way this is done is somewhat hacky - the closure signature is appended to the args,
//! as well as the closure kind "encoded" as a type. Also, special handling is needed when
//! the closure signature contains a reference to the original closure.
//!
//! Note that you should be careful not to allow the output of freshening to leak to the user in
//! error messages or in any other form. Freshening is only really useful as an internal detail.
//!
//! Because of the manipulation required to handle closures, doing arbitrary operations on
//! freshened types is not recommended. However, in addition to doing equality/hash
//! comparisons (for caching), it is possible to do a `ty::_match` operation between
//! two freshened types - this works even with the closure encoding.
//!
//! __An important detail concerning regions.__ The freshener also replaces *all* free regions with
//! 'erased. The reason behind this is that, in general, we do not take region relationships into
//! account when making type-overloaded decisions. This is important because of the design of the
//! region inferencer, which is not based on unification but rather on accumulating and then
//! solving a set of constraints. In contrast, the type inferencer assigns a value to each type
//! variable only once, and it does so as soon as it can, so it is reasonable to ask what the type
//! inferencer knows "so far".

use rustc_middle::bug;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
};

use super::InferCtxt;

pub struct TypeFreshener<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> TypeFreshener<'a, 'tcx> {
        TypeFreshener { infcx }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for TypeFreshener<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            // Leave bound regions alone, since they affect selection via the leak check.
            ty::ReBound(..) => r,
            // Leave error regions alone, since they affect selection b/c of incompleteness.
            ty::ReError(_) => r,

            ty::ReEarlyParam(..)
            | ty::ReLateParam(_)
            | ty::ReVar(_)
            | ty::RePlaceholder(..)
            | ty::ReStatic
            | ty::ReErased => self.cx().lifetimes.re_erased,
        }
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_infer() && !t.has_erasable_regions() {
            t
        } else {
            match *t.kind() {
                ty::Infer(v) => self.fold_infer_ty(v).unwrap_or(t),

                // This code is hot enough that a non-debug assertion here makes a noticeable
                // difference on benchmarks like `wg-grammar`.
                #[cfg(debug_assertions)]
                ty::Placeholder(..) | ty::Bound(..) => bug!("unexpected type {:?}", t),

                _ => t.super_fold_with(self),
            }
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match ct.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(v)) => {
                let input =
                    self.infcx.inner.borrow_mut().const_unification_table().probe_value(v).known();
                match input {
                    Some(ct) => ct.fold_with(self),
                    None => self.infcx.tcx.consts.fresh_const,
                }
            }
            ty::ConstKind::Infer(ty::InferConst::Fresh) => ct,

            ty::ConstKind::Bound(..) | ty::ConstKind::Placeholder(_) => {
                bug!("unexpected const {:?}", ct)
            }

            ty::ConstKind::Param(_)
            | ty::ConstKind::Value(_)
            | ty::ConstKind::Unevaluated(..)
            | ty::ConstKind::Expr(..)
            | ty::ConstKind::Error(_) => ct.super_fold_with(self),
        }
    }
}

impl<'a, 'tcx> TypeFreshener<'a, 'tcx> {
    // This is separate from `fold_ty` to keep that method small and inlinable.
    #[inline(never)]
    fn fold_infer_ty(&mut self, v: ty::InferTy) -> Option<Ty<'tcx>> {
        match v {
            ty::TyVar(v) => {
                let value = self.infcx.inner.borrow_mut().type_variables().probe(v).known();
                Some(match value {
                    Some(ty) => ty.fold_with(self),
                    None => self.infcx.tcx.types.fresh_ty,
                })
            }

            ty::IntVar(v) => {
                let value = self.infcx.inner.borrow_mut().int_unification_table().probe_value(v);
                Some(match value {
                    ty::IntVarValue::IntType(ty) => Ty::new_int(self.infcx.tcx, ty).fold_with(self),
                    ty::IntVarValue::UintType(ty) => {
                        Ty::new_uint(self.infcx.tcx, ty).fold_with(self)
                    }
                    ty::IntVarValue::Unknown => self.infcx.tcx.types.fresh_int_ty,
                })
            }

            ty::FloatVar(v) => {
                let value = self.infcx.inner.borrow_mut().float_unification_table().probe_value(v);
                Some(match value {
                    ty::FloatVarValue::Known(ty) => {
                        Ty::new_float(self.infcx.tcx, ty).fold_with(self)
                    }
                    ty::FloatVarValue::Unknown => self.infcx.tcx.types.fresh_float_ty,
                })
            }

            ty::FreshTy | ty::FreshIntTy | ty::FreshFloatTy => None,
        }
    }
}
