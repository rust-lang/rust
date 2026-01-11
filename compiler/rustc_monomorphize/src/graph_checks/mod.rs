//! Checks that need to operate on the entire mono item graph
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::TyCtxt;

use crate::collector::UsageMap;
use crate::graph_checks::statics::check_static_initializers_are_acyclic;

mod statics;

pub(super) fn target_specific_checks<'tcx, 'a, 'b>(
    tcx: TyCtxt<'tcx>,
    mono_items: &'a [MonoItem<'tcx>],
    usage_map: &'b UsageMap<'tcx>,
) {
    if tcx.sess.target.options.static_initializer_must_be_acyclic {
        check_static_initializers_are_acyclic(tcx, mono_items, usage_map);
    }
}
