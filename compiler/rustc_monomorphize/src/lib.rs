// tidy-alphabetical-start
#![feature(file_buffered)]
#![feature(impl_trait_in_assoc_type)]
#![feature(once_cell_get_mut)]
// tidy-alphabetical-end

use rustc_hir::lang_items::LangItem;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::adjustment::CustomCoerceUnsized;
use rustc_middle::ty::{self, Ty};
use rustc_middle::util::Providers;
use rustc_middle::{bug, traits};
use rustc_span::ErrorGuaranteed;

mod cast_sensitivity;
mod collector;
mod erasure_safe;
mod errors;
mod graph_checks;
mod mono_checks;
mod partitioning;
mod resolved_bodies;
mod table_layout;
mod trait_cast_requests;
mod trait_graph;
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
    providers.queries.gather_trait_cast_requests = trait_cast_requests::gather_trait_cast_requests;
    providers.queries.trait_cast_graph = trait_graph::trait_cast_graph;
    providers.queries.outlives_reachability = trait_graph::outlives_reachability;
    providers.queries.impl_universally_admissible = trait_graph::impl_universally_admissible;
    providers.queries.trait_cast_layout = table_layout::trait_cast_layout;
    providers.queries.trait_cast_table = table_layout::trait_cast_table;
    providers.queries.trait_cast_table_alloc = table_layout::trait_cast_table_alloc;
    providers.queries.global_crate_id_alloc = table_layout::global_crate_id_alloc;
    providers.queries.augmented_outlives_for_call = cast_sensitivity::augmented_outlives_for_call;
    providers.queries.is_lifetime_erasure_safe = erasure_safe::is_lifetime_erasure_safe;
}
