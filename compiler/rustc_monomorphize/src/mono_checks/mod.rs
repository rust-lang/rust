//! This implements a single query, `check_mono_fn`, that gets fired for each
//! monomorphization of all functions. This lets us implement monomorphization-time
//! checks in a way that is friendly to incremental compilation.

use rustc_middle::query::Providers;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::ErrorGuaranteed;

mod abi_check;
mod move_check;

fn check_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    let body = tcx.instance_mir(instance.def);
    // Run both checks unconditionally for their diagnostic side effects before combining. Each
    // returns `Some` if it emitted a hard post-monomorphization error: `abi_check` for ABI errors,
    // and `move_check` when `large_assignments` is escalated to `deny`/`forbid`.
    let abi_res = abi_check::check_feature_dependent_abi(tcx, instance, body);
    let move_res = move_check::check_moves(tcx, instance, body);
    match abi_res.or(move_res) {
        Some(guar) => Err(guar),
        None => Ok(()),
    }
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mono_item, ..*providers }
}
