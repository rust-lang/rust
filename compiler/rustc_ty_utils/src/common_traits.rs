//! Queries for checking whether a type implements one of a few common traits.

use rustc_hir::lang_items::LangItem;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_trait_selection::traits;

fn is_copy_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, LangItem::Copy)
}

fn is_sized_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, LangItem::Sized)
}

fn is_freeze_raw<'tcx>(tcx: TyCtxt<'tcx>, query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
    is_item_raw(tcx, query, LangItem::Freeze)
}

fn is_item_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    query: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    item: LangItem,
) -> bool {
    let (param_env, ty) = query.into_parts();
    let trait_def_id = tcx.require_lang_item(item, None);
    tcx.infer_ctxt().enter(|infcx| {
        traits::type_known_to_meet_bound_modulo_regions(
            &infcx,
            param_env,
            ty,
            trait_def_id,
            DUMMY_SP,
        )
    })
}

pub(crate) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { is_copy_raw, is_sized_raw, is_freeze_raw, ..*providers };
}
