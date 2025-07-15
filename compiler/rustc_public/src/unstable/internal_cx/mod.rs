//! Implementation of InternalCx.

pub(crate) use helpers::*;
use rustc_middle::ty::{List, Ty, TyCtxt};
use rustc_middle::{mir, ty};

use super::InternalCx;

pub(crate) mod helpers;

impl<'tcx, T: InternalCx<'tcx>> ExistentialProjectionHelpers<'tcx> for T {
    fn new_from_args(
        &self,
        def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgsRef<'tcx>,
        term: ty::Term<'tcx>,
    ) -> ty::ExistentialProjection<'tcx> {
        ty::ExistentialProjection::new_from_args(self.tcx(), def_id, args, term)
    }
}

impl<'tcx, T: InternalCx<'tcx>> ExistentialTraitRefHelpers<'tcx> for T {
    fn new_from_args(
        &self,
        trait_def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> ty::ExistentialTraitRef<'tcx> {
        ty::ExistentialTraitRef::new_from_args(self.tcx(), trait_def_id, args)
    }
}

impl<'tcx, T: InternalCx<'tcx>> TraitRefHelpers<'tcx> for T {
    fn new_from_args(
        &self,
        trait_def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> ty::TraitRef<'tcx> {
        ty::TraitRef::new_from_args(self.tcx(), trait_def_id, args)
    }
}

impl<'tcx> InternalCx<'tcx> for TyCtxt<'tcx> {
    fn tcx(self) -> TyCtxt<'tcx> {
        self
    }

    fn lift<T: ty::Lift<TyCtxt<'tcx>>>(self, value: T) -> Option<T::Lifted> {
        TyCtxt::lift(self, value)
    }

    fn mk_args_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: ty::CollectAndApply<ty::GenericArg<'tcx>, ty::GenericArgsRef<'tcx>>,
    {
        TyCtxt::mk_args_from_iter(self, iter)
    }

    fn mk_pat(self, v: ty::PatternKind<'tcx>) -> ty::Pattern<'tcx> {
        TyCtxt::mk_pat(self, v)
    }

    fn mk_poly_existential_predicates(
        self,
        eps: &[ty::PolyExistentialPredicate<'tcx>],
    ) -> &'tcx List<ty::PolyExistentialPredicate<'tcx>> {
        TyCtxt::mk_poly_existential_predicates(self, eps)
    }

    fn mk_type_list(self, v: &[Ty<'tcx>]) -> &'tcx List<Ty<'tcx>> {
        TyCtxt::mk_type_list(self, v)
    }

    fn lifetimes_re_erased(self) -> ty::Region<'tcx> {
        self.lifetimes.re_erased
    }

    fn mk_bound_variable_kinds_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: ty::CollectAndApply<ty::BoundVariableKind, &'tcx List<ty::BoundVariableKind>>,
    {
        TyCtxt::mk_bound_variable_kinds_from_iter(self, iter)
    }

    fn mk_place_elems(self, v: &[mir::PlaceElem<'tcx>]) -> &'tcx List<mir::PlaceElem<'tcx>> {
        TyCtxt::mk_place_elems(self, v)
    }

    fn adt_def(self, def_id: rustc_hir::def_id::DefId) -> ty::AdtDef<'tcx> {
        self.adt_def(def_id)
    }
}
