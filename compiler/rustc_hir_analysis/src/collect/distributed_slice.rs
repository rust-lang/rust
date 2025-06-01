use rand::SeedableRng;
use rand::seq::SliceRandom;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{DistributedSlice, ItemKind};
use rustc_middle::middle::distributed_slice::DistributedSliceAddition;
use rustc_middle::ty::TyCtxt;

pub(super) fn distributed_slice_elements<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> DefIdMap<Vec<DistributedSliceAddition>> {
    let mut slice_elements = DefIdMap::<Vec<DistributedSliceAddition>>::default();

    for i in tcx.hir_free_items() {
        let addition_def_id = i.owner_id.def_id;
        if let ItemKind::Const(.., DistributedSlice::Addition(declaration_def_id)) =
            tcx.hir_expect_item(addition_def_id).kind
        {
            slice_elements
                .entry(declaration_def_id.to_def_id())
                .or_default()
                .push(DistributedSliceAddition::Single(addition_def_id));
        }

        if let ItemKind::Const(.., DistributedSlice::AdditionMany(declaration_def_id, _)) =
            tcx.hir_expect_item(addition_def_id).kind
        {
            slice_elements
                .entry(declaration_def_id.to_def_id())
                .or_default()
                .push(DistributedSliceAddition::Many(addition_def_id));
        }
    }

    let mut res = DefIdMap::<Vec<DistributedSliceAddition>>::default();

    for (key, mut registered_values) in
        tcx.with_stable_hashing_context(|hcx| slice_elements.into_sorted(&hcx, true))
    {
        // shuffle seeded by the defpathhash of the registry
        let item_seed = tcx.def_path_hash(key).0.to_smaller_hash();
        let mut rng = rand_xoshiro::Xoshiro128StarStar::seed_from_u64(item_seed.as_u64());
        registered_values.as_mut_slice().shuffle(&mut rng);

        res.insert(key, registered_values);
    }

    res
}
