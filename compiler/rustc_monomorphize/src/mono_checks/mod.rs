//! This implements a single query, `check_mono_fn`, that gets fired for each
//! monomorphization of all functions. This lets us implement monomorphization-time
//! checks in a way that is friendly to incremental compilation.

use rustc_middle::query::Providers;
use rustc_middle::ty::{Instance, InstanceKind, TyCtxt};

mod abi_check;
mod move_check;

fn check_mono_item<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let body = tcx.instance_mir(instance.def);
    abi_check::check_feature_dependent_abi(tcx, instance, body);
    move_check::check_moves(tcx, instance, body);
    if let InstanceKind::Item(def_id) = instance.def {
        if tcx.instantiate_and_check_impossible_predicates((def_id, instance.args)) {
            tcx.dcx().span_err(tcx.def_span(def_id), "post-mono");
        }
    }
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers {
        check_mono_item,
        skip_move_check_fns: move_check::skip_move_check_fns,
        ..*providers
    }
}
