// tidy-alphabetical-start
#![feature(box_patterns)]
#![feature(file_buffered)]
#![feature(impl_trait_in_assoc_type)]
#![feature(once_cell_get_mut)]
// tidy-alphabetical-end

use rustc_hir::lang_items::LangItem;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::adjustment::CustomCoerceUnsized;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_middle::{bug, traits};
use rustc_span::ErrorGuaranteed;

mod collector;
mod dead_fn_elim;
mod diagnostics;
mod graph_checks;
mod mono_checks;
mod partitioning;
mod used_set;
mod util;

fn custom_coerce_unsize_info<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    source_ty: Ty<'tcx>,
    target_ty: Ty<'tcx>,
) -> Result<CustomCoerceUnsized, ErrorGuaranteed> {
    let trait_ref = ty::TraitRef::new(
        tcx.tcx,
        tcx.require_lang_item(LangItem::CoerceUnsized, tcx.span),
        [source_ty, target_ty],
    );

    match tcx
        .codegen_select_candidate(ty::TypingEnv::fully_monomorphized().as_query_input(trait_ref))
    {
        Ok(traits::ImplSource::UserDefined(traits::ImplSourceUserDefinedData {
            impl_def_id,
            ..
        })) => Ok(tcx.coerce_unsized_info(*impl_def_id)?.custom_kind.unwrap()),
        impl_source => {
            bug!(
                "invalid `CoerceUnsized` from {source_ty} to {target_ty}: impl_source: {:?}",
                impl_source
            );
        }
    }
}

pub fn provide(providers: &mut Providers) {
    partitioning::provide(providers);
    mono_checks::provide(&mut providers.queries);
}

/// If `-Zdead-fn-emit-used-set=<dir>` is set, walk this crate's MIR (post-analysis, before
/// codegen) and write the per-dependency used-set files. Called from the analysis pass so it
/// works on `--emit=metadata` builds (no codegen required) — the first-build mechanism.
pub fn emit_used_set_if_requested(tcx: TyCtxt<'_>) {
    if let Some(dir) = &tcx.sess.opts.unstable_opts.dead_fn_emit_used_set {
        used_set::emit_used_sets(tcx, dir);
    }
}
