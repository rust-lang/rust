use rustc_middle::bug;
use rustc_middle::ty::trait_cast::IntrinsicSiteKind;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::sym;

/// Classifies an augmented intrinsic Instance by projecting
/// fully-monomorphized types from its generic args. Pure O(1) function.
///
/// Not a query: the computation is cheaper than query machinery overhead
/// (key hashing, dep-node creation, result storage), and all callers are
/// in `rustc_monomorphize`.
pub(crate) fn classify_intrinsic_site<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> IntrinsicSiteKind<'tcx> {
    let def_id = instance.def_id();
    let args = instance.args;
    let intrinsic = tcx
        .intrinsic(def_id)
        .unwrap_or_else(|| bug!("classify_intrinsic_site called on non-intrinsic: {:?}", instance));

    match intrinsic.name {
        s if s == sym::trait_metadata_index => {
            // Generic args: [Super, Sub, ...Outlives]
            IntrinsicSiteKind::Index {
                super_trait: args[0].expect_ty(),
                sub_trait: args[1].expect_ty(),
            }
        }
        s if s == sym::trait_metadata_table => {
            // Generic args: [Super, Concrete]
            IntrinsicSiteKind::Table {
                super_trait: args[0].expect_ty(),
                concrete_type: args[1].expect_ty(),
            }
        }
        s if s == sym::trait_metadata_table_len => {
            // Generic args: [Super]
            IntrinsicSiteKind::TableLen { super_trait: args[0].expect_ty() }
        }
        s if s == sym::trait_cast_is_lifetime_erasure_safe => {
            // Generic args: [Super, Tgt, ...Outlives]
            IntrinsicSiteKind::ErasureSafe {
                super_trait: args[0].expect_ty(),
                target_trait: args[1].expect_ty(),
            }
        }
        _ => {
            rustc_middle::bug!("classify_intrinsic_site: not a trait-cast intrinsic: {:?}", def_id)
        }
    }
}

/// Query provider: collect all augmented intrinsic Instances from all crates
/// and classify them into `TraitCastRequests`.
///
/// Iterates over `delayed_codegen_requests` for each crate (local and
/// upstream), extracts intrinsic callees from each delayed instance, and
/// classifies each intrinsic Instance into the appropriate
/// `IntrinsicSiteKind`.
pub(crate) fn gather_trait_cast_requests<'tcx>(
    tcx: TyCtxt<'tcx>,
    (): (),
) -> rustc_middle::ty::trait_cast::TraitCastRequests<'tcx> {
    use std::iter;

    use rustc_hir::def_id::LOCAL_CRATE;
    use rustc_middle::ty::trait_cast::TraitCastRequests;

    if !tcx.is_global_crate() {
        return TraitCastRequests::default();
    }

    let mut requests = TraitCastRequests::default();

    // Collect intrinsic Instances from all crates.
    // Local crate proxies into collect_and_partition_mono_items;
    // upstream crates decode from metadata.
    for &cnum in iter::once(&LOCAL_CRATE).chain(tcx.crates(())) {
        let delayed_list = tcx.delayed_codegen_requests(cnum);
        for delayed in delayed_list {
            for &intrinsic in delayed.intrinsic_callees {
                let site = classify_intrinsic_site(tcx, intrinsic);
                requests.add(site, intrinsic);
            }
        }
    }

    requests
}
