use rustc_hir::def_id::{CrateNum, LocalDefId, LOCAL_CRATE};

use std::cell::{Ref, RefCell, RefMut};

use super::wfcheck;
use crate::check::CheckItemTypesVisitor;
use crate::{ty, TyCtxt};

/// A wrapper for `InferCtxt`'s `in_progress_typeck_results` field.
#[derive(Copy, Clone)]
pub(super) struct MaybeInProgressTables<'a, 'tcx> {
    pub(super) maybe_typeck_results: Option<&'a RefCell<ty::TypeckResults<'tcx>>>,
}

impl<'a, 'tcx> MaybeInProgressTables<'a, 'tcx> {
    pub(super) fn borrow(self) -> Ref<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow() with no typeck results"
            ),
        }
    }

    pub(super) fn borrow_mut(self) -> RefMut<'a, ty::TypeckResults<'tcx>> {
        match self.maybe_typeck_results {
            Some(typeck_results) => typeck_results.borrow_mut(),
            None => bug!(
                "MaybeInProgressTables: inh/fcx.typeck_results.borrow_mut() with no typeck results"
            ),
        }
    }
}

pub(super) fn check_mod_item_types(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut CheckItemTypesVisitor { tcx });
}

pub(super) fn check_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_item_well_formed(tcx, def_id);
}

pub(super) fn check_trait_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_trait_item(tcx, def_id);
}

pub(super) fn check_impl_item_well_formed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    wfcheck::check_impl_item(tcx, def_id);
}

pub(super) fn typeck_item_bodies(tcx: TyCtxt<'_>, crate_num: CrateNum) {
    debug_assert!(crate_num == LOCAL_CRATE);
    tcx.par_body_owners(|body_owner_def_id| {
        tcx.ensure().typeck(body_owner_def_id);
    });
}
