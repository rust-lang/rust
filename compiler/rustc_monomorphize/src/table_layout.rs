use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitMatrix, BitRowRef, DenseBitSet};
use rustc_middle::bug;
use rustc_middle::mir::Mutability;
use rustc_middle::mir::interpret::{AllocId, AllocInit, Allocation, Pointer, Scalar, alloc_range};
use rustc_middle::ty::trait_cast::{
    FingerprintedTy, IntrinsicResolutions, OutlivesClass, SlotInfo, TableLayout, TraitCastRequests,
};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeVisitable};

use crate::erasure_safe::{
    collect_all_binder_vars, root_exposed_target_bvs, trait_metadata_index_outlives_class,
};
use crate::trait_graph::{
    RegionPositionCollector, extract_dyn_bv_positions, extract_impl_trait_ref_regions,
    region_to_bvs, resolve_dyn_satisfaction,
};

/// Assign table slot indices for all (sub_trait, outlives_class) pairs
/// in the trait cast graph, applying condensation to collapse classes
/// that admit identical sets of implementations.
///
/// Sub-traits and concrete types are processed in a deterministic order
/// (see `FingerprintedTy`) to ensure reproducible table layouts across
/// compilations.
///
/// **Fast path:** When every resolved impl (for participating concrete
/// types) is universally admissible (`impl_universally_admissible`), all
/// outlives classes for that sub-trait are equivalent and collapse to a
/// single slot — no per-class admissibility check needed.
///
/// **Full condensation:** Otherwise, `condense_outlives_classes` builds
/// a `BitMatrix` of (class × concrete_type) admissibility and groups
/// classes with identical rows into condensation groups, each getting
/// one shared slot.
pub(crate) fn trait_cast_layout<'tcx>(tcx: TyCtxt<'tcx>, root: Ty<'tcx>) -> TableLayout<'tcx> {
    assert!(root.is_known_rigid(), "trait cast root super-trait must be monomorphized: {root}");
    let graph = tcx.trait_cast_graph(root);
    let mut index_map: UnordMap<(Ty<'tcx>, OutlivesClass<'tcx>), usize> = UnordMap::default();
    let mut slot_info: Vec<SlotInfo<'tcx>> = Vec::new();
    let mut next_index: usize = 0;

    // Deterministic iteration: sort sub-trait keys by fingerprint via
    // FingerprintedTy's StableCompare impl.
    let sub_trait_pairs = graph
        .sub_traits
        .items()
        .map(|(k, v)| (*k, v))
        .into_sorted_stable_ord_by_key(|item| &item.0);

    // Materialize concrete types in deterministic order (shared across
    // all sub-traits in this root). The column order in the condensation
    // BitMatrix must be stable so that row patterns are reproducible.
    let concrete_types_sorted: Vec<FingerprintedTy<'tcx>> =
        graph.concrete_types.items().copied().into_sorted_stable_ord();

    for (sub_trait_fp, info) in &sub_trait_pairs {
        let sub_trait: Ty<'tcx> = **sub_trait_fp;

        if info.outlives_classes.is_empty() {
            continue;
        }

        let ty::Dynamic(dyn_data, ..) = sub_trait.kind() else {
            bug!("trait_cast_layout: sub_trait {sub_trait:?} is not a dyn type");
        };
        let num_bvs = collect_all_binder_vars(tcx, dyn_data).total_count();

        // --- Resolve impls for participating concrete types (shared) ---
        //
        // Needed by both the fast path and full condensation, so computed
        // once up front. Only impls for concrete types that actually
        // participate in the program's monomorphization set are considered
        // — avoiding dep-graph pollution from `all_impls`.
        let impl_cache: Vec<Option<DefId>> = concrete_types_sorted
            .iter()
            .map(|ty| resolve_dyn_satisfaction(tcx, **ty, sub_trait))
            .collect();

        // --- Fast path: all participating impls universally admissible ---
        //
        // When every resolved impl has: (a) no concrete lifetimes in the
        // trait ref, (b) no param aliasing, (c) no RegionOutlives
        // where-clauses on trait lifetime params, and (d) no shared
        // Self/trait params — all classes are equivalent and collapse to
        // one slot.
        let all_admissible = impl_cache
            .iter()
            .filter_map(|&r| r)
            .all(|impl_def_id| tcx.impl_universally_admissible(impl_def_id));

        // Materialize classes in StableOrd order for deterministic indices.
        let classes: Vec<OutlivesClass<'tcx>> =
            info.outlives_classes.items().copied().into_sorted_stable_ord();

        if all_admissible {
            let slot = next_index;
            next_index += 1;
            // All classes collapse to one slot. The representative is
            // the first (minimum by StableOrd) — universal admissibility
            // means the choice has no semantic effect, but the
            // representative must be reproducible across runs.
            let representative = classes[0];
            for &class in &classes {
                index_map.insert((sub_trait, class), slot);
            }
            slot_info.push(SlotInfo { sub_trait, outlives_class: representative, num_bvs });
            continue;
        }

        // --- Full condensation ---
        //
        // Compute per-class admissibility vectors over concrete_types,
        // group classes with identical vectors, assign one slot per
        // group. Reuses `impl_cache` from above.
        let condensed_groups =
            condense_outlives_classes(tcx, root, sub_trait, &classes, &impl_cache, num_bvs);

        for (group, repr_class) in condensed_groups {
            let slot = next_index;
            next_index += 1;
            for class_idx in group.iter() {
                let class = classes[class_idx as usize];
                index_map.insert((sub_trait, class), slot);
            }
            slot_info.push(SlotInfo { sub_trait, outlives_class: repr_class, num_bvs });
        }
    }

    TableLayout { root, table_length: next_index, index_map, slot_info }
}

/// Returns groups of class indices that are equivalent (produce
/// identical admissibility vectors across all concrete types).
/// Class indices are positions in the `classes` slice.
///
/// **Determinism contract:** `classes` must be in `StableOrd` order
/// and `impl_cache` columns must follow the deterministic
/// `concrete_types_sorted` order. Groups are returned sorted by their
/// minimum class index, so the first group always contains class 0,
/// ensuring reproducible slot assignment across compilations.
///
/// `impl_cache` is the pre-computed per-concrete-type impl resolution
/// from `trait_cast_layout` — shared with the fast path to avoid
/// redundant trait-solver queries. Its column order matches
/// `concrete_types_sorted`.
fn condense_outlives_classes<'tcx>(
    tcx: TyCtxt<'tcx>,
    root: Ty<'tcx>,
    sub_trait: Ty<'tcx>,
    classes: &[OutlivesClass<'tcx>],
    impl_cache: &[Option<DefId>],
    num_bvs: usize,
) -> Vec<(DenseBitSet<u32>, OutlivesClass<'tcx>)> {
    let num_types = impl_cache.len();
    let num_classes = classes.len();

    // Flat BitMatrix: rows = classes, columns = concrete types.
    // One bit per (class, concrete_type) pair, set if the impl is
    // admissible under that class. Single allocation.
    let mut matrix: BitMatrix<u32, u32> = BitMatrix::new(num_classes, num_types);

    let dim = num_bvs + 1;
    for (class_idx, class) in classes.iter().enumerate() {
        // Reachability matrix from the cached query — shared with
        // the population query and the erasure-safe check.
        let reachability = tcx.outlives_reachability((class.entries, dim));
        for (type_idx, maybe_impl) in impl_cache.iter().enumerate() {
            if let Some(impl_def_id) = maybe_impl {
                if impl_admissible_under_class(
                    tcx,
                    *impl_def_id,
                    root,
                    sub_trait,
                    &reachability,
                    num_bvs,
                ) {
                    matrix.insert(class_idx as u32, type_idx as u32);
                }
            }
        }
    }

    // Group classes by identical admissibility rows.
    // row_ref() returns a BitRowRef whose Eq/Hash masks out excess
    // bits in the final word.
    //
    // Because `classes` is sorted by `StableOrd` and we iterate
    // `0..num_classes`, the FxIndexMap insertion order is
    // deterministic: each novel row pattern is first encountered at
    // the smallest class_idx that produces it.
    let mut groups: FxIndexMap<BitRowRef<'_>, DenseBitSet<u32>> = FxIndexMap::default();
    for class_idx in 0..num_classes {
        groups
            .entry(matrix.row_ref(class_idx as u32))
            .or_insert_with(|| DenseBitSet::new_empty(num_classes))
            .insert(class_idx as u32);
    }

    // Return groups sorted by minimum class index within each group.
    // Since `classes` is StableOrd-sorted, this is equivalent to
    // sorting by the smallest `OutlivesClass` representative — making
    // slot assignment deterministic.
    //
    // Each group is paired with its representative outlives class
    // (smallest class index). The population query calls
    // `outlives_reachability` with this class's entries to obtain
    // the reachability matrix on demand (cached by the query system).
    let mut result: Vec<(DenseBitSet<u32>, OutlivesClass<'tcx>)> = groups
        .into_values()
        .map(|group| {
            let repr_idx = group.iter().next().unwrap() as usize;
            (group, classes[repr_idx])
        })
        .collect();
    result.sort_by_key(|(group, _)| group.iter().next().unwrap());
    result
}

/// Check whether an impl is admissible under a specific outlives class
/// for a given dyn type (cast target).
///
/// Algorithm:
/// 1. Extract bv indices from the dyn type's binder.
/// 2. Walk the impl's trait ref in parallel to build a param→bv mapping.
///    2b. Self-anchored parameters require 'static-equivalence unless
///    their mapped bvs are root-exposed.
/// 3. If one impl param maps to multiple distinct bvs, those bvs must
///    be equivalent under the class.
/// 4. Explicit RegionOutlives where clauses must be implied by the class.
///
/// Uses the pre-computed `reachability` matrix (from `outlives_reachability`)
/// for O(1) outlives lookups. Index `num_bvs` in the matrix represents
/// `'static`.
pub(crate) fn impl_admissible_under_class<'tcx>(
    tcx: TyCtxt<'tcx>,
    impl_def_id: DefId,
    root_dyn_type: Ty<'tcx>,
    dyn_type: Ty<'tcx>,
    reachability: &BitMatrix<usize, usize>,
    num_bvs: usize,
) -> bool {
    let Some(impl_trait_ref) = tcx.impl_opt_trait_ref(impl_def_id) else {
        bug!("impl_admissible_under_class: impl {impl_def_id:?} has no trait ref");
    };
    let impl_trait_ref = impl_trait_ref.skip_binder();

    // O(1) reachability lookup. 'static is at index num_bvs (= dim - 1).
    let remap = |idx: usize| if idx == usize::MAX { num_bvs } else { idx };
    let implies =
        |longer: usize, shorter: usize| reachability.contains(remap(longer), remap(shorter));

    // Walk the dyn type's existential trait ref to get the bound
    // variable index at each trait-lifetime position. Positions where
    // the dyn type has a concrete lifetime (e.g., 'static embedded in
    // a type arg) get `None`.
    let dyn_bvs: Vec<Option<usize>> = extract_dyn_bv_positions(tcx, dyn_type);

    // At each position, pair the dyn type's bv index with the impl's
    // region to build the mapping.
    let impl_regions: Vec<ty::Region<'tcx>> = extract_impl_trait_ref_regions(tcx, impl_trait_ref);

    assert_eq!(
        dyn_bvs.len(),
        impl_regions.len(),
        "dyn type and impl must have same number of trait lifetime positions"
    );

    // Build: impl_param → set of bv indices it maps to (deduplicated).
    let mut param_to_bvs: FxHashMap<RegionVid, DenseBitSet<usize>> = FxHashMap::default();

    for (&dyn_bv, &impl_region) in dyn_bvs.iter().zip(impl_regions.iter()) {
        let Some(bv) = dyn_bv else {
            // Dyn type has a concrete lifetime at this position.
            // The impl's region at this position must be compatible.
            // (Handled separately — see below.)
            continue;
        };

        match impl_region.kind() {
            // Free impl param — record which bvs it maps to.
            ty::ReEarlyParam(param) => {
                param_to_bvs
                    .entry(RegionVid::from_u32(param.index))
                    .or_insert_with(|| DenseBitSet::new_empty(num_bvs))
                    .insert(bv);
            }
            // Impl fixes this position to 'static.
            // The class must imply this bv outlives 'static.
            ty::ReStatic => {
                if !implies(bv, usize::MAX) {
                    return false;
                }
            }
            // Higher-ranked in impl — not a free param, skip.
            ty::ReBound(..) => {}
            // Conservative reject for other concrete lifetimes.
            _ => {
                return false;
            }
        }
    }

    // Self-anchored parameters: walk the impl's Self type to detect
    // impl params that appear in BOTH Self and trait-arg positions.
    // Such params are anchored to the concrete type's (erased)
    // lifetime. If the mapped target bvs are structurally exposed
    // through the root supertrait, the erasure-safe check validates
    // that root<->target correspondence at the cast site, so no extra
    // table-side restriction is needed here. Non-root-exposed shared
    // bvs remain hidden behind Self and therefore still require the
    // conservative 'static-equivalence check.
    let root_exposed_bvs = root_exposed_target_bvs(tcx, root_dyn_type, dyn_type);
    let self_ty = impl_trait_ref.self_ty();
    let mut self_region_collector = RegionPositionCollector::new();
    self_ty.visit_with(&mut self_region_collector);
    for self_region in self_region_collector.into_regions() {
        if let ty::ReEarlyParam(param) = self_region.kind() {
            if let Some(bvs) = param_to_bvs.get(&RegionVid::from_u32(param.index)) {
                if bvs.iter().all(|bv| root_exposed_bvs.contains(bv)) {
                    continue;
                }
                // Shared param: appears in Self AND trait args.
                // All mapped bvs must be 'static-equivalent.
                for bv in bvs.iter() {
                    if !implies(bv, usize::MAX) || !implies(usize::MAX, bv) {
                        return false;
                    }
                }
            }
        }
    }

    // Parameter aliasing: if one impl param maps to multiple DISTINCT
    // bvs, those bvs must be equivalent (mutual outlives) under the
    // class. If they map to the same bv (due to dyn type aliasing), no
    // constraint is needed — the dyn binder already guarantees it.
    //
    // Iteration order is irrelevant: the result is a pure conjunction
    // over all params — each param's check is independent.
    #[allow(rustc::potential_query_instability)]
    for (_param, bvs) in &param_to_bvs {
        let mut iter = bvs.iter();
        let Some(first) = iter.next() else { continue };
        for bv in iter {
            if !implies(first, bv) || !implies(bv, first) {
                return false;
            }
        }
    }

    // Explicit RegionOutlives where clauses.
    let predicates = tcx.predicates_of(impl_def_id);
    'preds: for (pred, _) in predicates.predicates {
        if let ty::ClauseKind::RegionOutlives(outlives) = pred.kind().skip_binder() {
            let longer_bvs = region_to_bvs(&param_to_bvs, outlives.0);
            let shorter_bvs = region_to_bvs(&param_to_bvs, outlives.1);

            match (longer_bvs, shorter_bvs) {
                (Some(ls), Some(ss)) => {
                    for l in ls.iter() {
                        for s in ss.iter() {
                            if l != s && !implies(l, s) {
                                return false;
                            }
                        }
                    }
                }
                (Some(ls), None) if outlives.1.is_static() => {
                    for l in ls.iter() {
                        if !implies(l, usize::MAX) {
                            return false;
                        }
                    }
                }
                (Some(ls), None) if !outlives.1.is_static() => {
                    // 'a: 'b where 'b is hidden. We can't prove anything about this.
                    // typeck will emit a diagnostic warning.
                    if ls.iter().all(|l| implies(l, usize::MAX)) {
                        continue 'preds;
                    }
                    return false;
                }
                (None, Some(_)) if !outlives.0.is_static() => {
                    // 'a: 'b where 'a is hidden. We can't prove anything about this.
                    // typeck will emit a diagnostic warning.
                    return false;
                }
                (None, Some(_)) if outlives.0.is_static() => {
                    // 'static: shorter — tautology.
                }
                _ => {
                    return false;
                }
            }
        }
    }

    true
}

/// Populate the trait cast table for a (root supertrait, concrete type) pair.
///
/// For each table slot (one per (sub_trait, outlives_class) pair from the
/// layout), determines whether the concrete type implements the sub-trait
/// under that slot's outlives class. If so, records the vtable `AllocId`;
/// otherwise the slot is `None`.
///
/// Groups slots by sub-trait to avoid redundant impl resolution — all slots
/// for the same sub-trait share the same `impl_def_id`.
pub(crate) fn trait_cast_table<'tcx>(
    tcx: TyCtxt<'tcx>,
    (root, concrete_type): (Ty<'tcx>, Ty<'tcx>),
) -> &'tcx [Option<AllocId>] {
    let layout = tcx.trait_cast_layout(root);
    let mut table: Vec<Option<AllocId>> = vec![None; layout.table_length];

    for sub_trait in layout.sub_traits() {
        let Some(impl_def_id) = resolve_dyn_satisfaction(tcx, concrete_type, sub_trait) else {
            // Concrete type does not implement this sub-trait (or fails
            // an auto trait bound). All slots for this sub-trait stay None.
            continue;
        };

        // Extract the existential trait ref for vtable_allocation.
        let ty::Dynamic(dyn_data, ..) = sub_trait.kind() else {
            bug!("trait_cast_table: sub_trait {sub_trait:?} is not a dyn type");
        };
        let sub_trait_ref = dyn_data.principal().map(|p| p.skip_binder());

        for (index, slot_info) in layout.slots_for_sub_trait(sub_trait) {
            let reachability = tcx
                .outlives_reachability((slot_info.outlives_class.entries, slot_info.num_bvs + 1));

            if impl_admissible_under_class(
                tcx,
                impl_def_id,
                root,
                sub_trait,
                &reachability,
                slot_info.num_bvs,
            ) {
                let vtable = tcx.vtable_allocation((concrete_type, sub_trait_ref));
                table[index] = Some(vtable);
            }
            // else: impl exists but not admissible under this class — None.
        }
    }

    tcx.arena.alloc_from_iter(table)
}

/// Build a static allocation holding the trait cast metadata table for a
/// (root supertrait, concrete type) pair. Each entry is a pointer-sized
/// slot: `Some(vtable_alloc_id)` becomes a pointer to that vtable;
/// `None` becomes a null pointer.
///
/// The resulting allocation is immutable and placed in `.rodata` (or
/// equivalent) by the codegen backend.
fn emit_table_static<'tcx>(tcx: TyCtxt<'tcx>, root: Ty<'tcx>, concrete_type: Ty<'tcx>) -> AllocId {
    let table = tcx.trait_cast_table((root, concrete_type));

    let ptr_size = tcx.data_layout.pointer_size();
    let ptr_align = tcx.data_layout.pointer_align().abi;
    let table_size = ptr_size * u64::try_from(table.len()).unwrap();
    let mut alloc = Allocation::new(table_size, ptr_align, AllocInit::Uninit, ());

    for (idx, entry) in table.iter().enumerate() {
        let idx: u64 = u64::try_from(idx).unwrap();
        let scalar = match entry {
            None => Scalar::from_maybe_pointer(Pointer::null(), &tcx),
            Some(vtable_alloc_id) => {
                let vptr = Pointer::from(*vtable_alloc_id);
                Scalar::from_pointer(vptr, &tcx)
            }
        };
        alloc
            .write_scalar(&tcx, alloc_range(ptr_size * idx, ptr_size), scalar)
            .expect("failed to build trait cast metadata table");
    }

    alloc.mutability = Mutability::Not;
    tcx.reserve_and_set_memory_alloc(tcx.mk_const_alloc(alloc))
}

/// Query provider: returns the `AllocId` of the metadata table static for
/// the given (root supertrait, concrete type) pair.
pub(crate) fn trait_cast_table_alloc<'tcx>(
    tcx: TyCtxt<'tcx>,
    (root, concrete_type): (Ty<'tcx>, Ty<'tcx>),
) -> AllocId {
    emit_table_static(tcx, root, concrete_type)
}

/// Create a single immutable `u8 = 0` static allocation whose address
/// serves as the unique global crate identifier. Only the address is
/// significant — the value is unspecified.
///
/// The allocation uses byte alignment and immutable mutability to ensure
/// it lands in `.rodata` (or equivalent). The codegen backend must not
/// mark it `unnamed_addr` — LLVM's default `named_addr` semantics
/// guarantee the address is preserved through optimization.
fn get_or_create_global_crate_id<'tcx>(tcx: TyCtxt<'tcx>) -> AllocId {
    let mut alloc = Allocation::from_bytes_byte_aligned_immutable(&[0u8], ());
    alloc.address_significant = true;
    tcx.reserve_and_set_memory_alloc(tcx.mk_const_alloc(alloc))
}

/// Query provider: returns the `AllocId` of the per-global-crate `u8`
/// static used for cross-crate trait-cast safety checks.
pub(crate) fn global_crate_id_alloc<'tcx>(tcx: TyCtxt<'tcx>, _: ()) -> AllocId {
    get_or_create_global_crate_id(tcx)
}

/// Build the lookup table that maps each table-dependent intrinsic to
/// its resolved constant value. Iterates over the classified requests,
/// delegates to the per-intrinsic resolution logic, and collects
/// results into an [`IntrinsicResolutions`] for use by
/// `cascade_canonicalize`.
///
/// By this point, `trait_cast_layout(root)` and
/// `trait_cast_table_alloc(root, concrete_type)` have already been forced
/// in `resolve_trait_cast_globals`'s query-driving loop. All query calls
/// below are therefore cache hits.
pub(crate) fn build_intrinsic_resolutions<'tcx>(
    tcx: TyCtxt<'tcx>,
    requests: &TraitCastRequests<'tcx>,
) -> IntrinsicResolutions<'tcx> {
    let global_crate_id = get_or_create_global_crate_id(tcx);

    // --- trait_metadata_index: (sub_trait, outlives_class) → slot index ---
    // Multiple augmented intrinsic Instances may request the same
    // (sub_trait, outlives_class) — e.g. the same intrinsic
    // monomorphized from different call sites in different crates.
    // Deduplication via `entry` makes the point-lookup-only contract
    // of IntrinsicResolutions explicit.
    let mut indices: UnordMap<(Ty<'tcx>, OutlivesClass<'tcx>), usize> = UnordMap::default();
    for req in &requests.index_requests {
        let outlives_class =
            trait_metadata_index_outlives_class(tcx, req.super_trait, req.sub_trait, req.instance);
        indices.entry((req.sub_trait, outlives_class)).or_insert_with(|| {
            let layout = tcx.trait_cast_layout(req.super_trait);
            *layout
                .index_map
                .get(&(req.sub_trait, outlives_class))
                .expect("index request for (sub_trait, outlives_class) not found in layout")
        });

        // Also populate indices for where-clause-derived classes in the
        // layout. These are added by `trait_cast_graph` and may not have
        // a corresponding mono request, but need to be accessible at
        // resolution time for callers that carry valid outlives evidence
        // through generic library code.
        let layout = tcx.trait_cast_layout(req.super_trait);
        for slot in &layout.slot_info {
            if slot.sub_trait == req.sub_trait {
                if let Some(&idx) = layout.index_map.get(&(slot.sub_trait, slot.outlives_class)) {
                    indices.entry((slot.sub_trait, slot.outlives_class)).or_insert(idx);
                }
            }
        }
    }

    // --- trait_metadata_table: (super_trait, concrete_type) → table static AllocId ---
    let mut tables: UnordMap<(Ty<'tcx>, Ty<'tcx>), AllocId> = UnordMap::default();
    let mut table_alloc_ids: Vec<AllocId> = Vec::new();
    for req in &requests.table_requests {
        tables.entry((req.super_trait, req.concrete_type)).or_insert_with(|| {
            let alloc_id = tcx.trait_cast_table_alloc((req.super_trait, req.concrete_type));
            table_alloc_ids.push(alloc_id);
            alloc_id
        });
    }

    // --- trait_metadata_table_len: super_trait → table length ---
    let mut table_lens: UnordMap<Ty<'tcx>, usize> = UnordMap::default();
    for req in &requests.table_len_requests {
        table_lens.entry(req.super_trait).or_insert_with(|| {
            let layout = tcx.trait_cast_layout(req.super_trait);
            layout.table_length
        });
    }

    IntrinsicResolutions { global_crate_id, indices, tables, table_lens, table_alloc_ids }
}
