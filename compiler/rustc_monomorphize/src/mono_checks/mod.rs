//! This implements a single query, `check_mono_fn`, that gets fired for each
//! monomorphization of all functions. This lets us implement monomorphization-time
//! checks in a way that is friendly to incremental compilation.

use rustc_middle::query::Providers;
use rustc_middle::ty::{Instance, TyCtxt};

mod abi_check;
mod move_check;

fn check_mono_item<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) {
    let body = tcx.instance_mir(instance.def);
    abi_check::check_feature_dependent_abi(tcx, instance, body);
    move_check::check_moves(tcx, instance, body);
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers {
        check_mono_item,
        skip_move_check_fns: move_check::skip_move_check_fns,
        ..*providers
    }
}
