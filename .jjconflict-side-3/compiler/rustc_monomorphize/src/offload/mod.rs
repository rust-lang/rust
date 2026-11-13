pub(crate) mod manifest;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::TyCtxt;

pub(crate) fn check_offload_kernels_instantiated<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_items: &[MonoItem<'tcx>],
) {
    let instantiated: FxHashSet<DefId> = mono_items
        .iter()
        .filter_map(|item| match item {
            MonoItem::Fn(instance) => Some(instance.def_id()),
            MonoItem::Static(def_id) => Some(*def_id),
            _ => None,
        })
        .collect();

    let crate_items = tcx.hir_crate_items(());
    let check = |def_id: DefId| {
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
            || !tcx.generics_of(def_id).requires_monomorphization(tcx)
            || !tcx.codegen_fn_attrs(def_id).flags.intersects(CodegenFnAttrFlags::OFFLOAD_KERNEL)
            || instantiated.contains(&def_id)
        {
            return;
        }
        tcx.dcx().emit_err(crate::diagnostics::GenericKernelNotInstantiated {
            span: tcx.def_span(def_id),
            def_path: tcx.def_path_str(def_id),
        });
    };
    for id in crate_items.free_items() {
        check(id.owner_id.to_def_id());
    }
    for id in crate_items.impl_items() {
        check(id.owner_id.to_def_id());
    }
    for id in crate_items.trait_items() {
        check(id.owner_id.to_def_id());
    }
}
