//! A set of traits that define a stable interface to rustc's internals.
//!
//! These traits are primarily used to clarify the behavior of different
//! functions that share the same name across various contexts.

use rustc_middle::ty;

pub(crate) trait ExistentialProjectionHelpers<'tcx> {
    fn new_from_args(
        &self,
        def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgs<'tcx>,
        term: ty::Term<'tcx>,
    ) -> ty::ExistentialProjection<'tcx>;
}

pub(crate) trait ExistentialTraitRefHelpers<'tcx> {
    fn new_from_args(
        &self,
        trait_def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgs<'tcx>,
    ) -> ty::ExistentialTraitRef<'tcx>;
}

pub(crate) trait TraitRefHelpers<'tcx> {
    fn new_from_args(
        &self,
        trait_def_id: rustc_span::def_id::DefId,
        args: ty::GenericArgs<'tcx>,
    ) -> ty::TraitRef<'tcx>;
}
