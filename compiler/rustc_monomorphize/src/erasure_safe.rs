//! Erasure-safe analysis for the `trait_cast_is_lifetime_erasure_safe`
//! intrinsic.
//!
//! Provides structural helpers (binder variable enumeration, supertrait
//! chain tracing) and the `resolve_erasure_safe_intrinsic` function +
//! `is_lifetime_erasure_safe` query for determining whether casting to a
//! target dyn type preserves lifetime identity.

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::bug;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::trait_cast::OutlivesClass;
use rustc_middle::ty::{self, GenericParamDefKind, Ty, TyCtxt, TypeVisitable, TypeVisitor};

// ── Types ──────────────────────────────────────────────────────────────────────

/// Binder variable info collected by TypeVisitor DFS over a dyn type's
/// existential predicates.
pub(crate) struct DynBinderVars {
    /// `(bv_index, PrincipalLocation)` for bvs in the principal
    /// `ExistentialTraitRef`.
    pub(crate) principal_entries: Vec<(usize, PrincipalLocation)>,
    /// `(bv_index, ProjectionLocation)` for bvs in `ExistentialProjection`s.
    pub(crate) projection_entries: Vec<(usize, ProjectionLocation)>,
}

impl DynBinderVars {
    /// Total number of distinct binder variables (max bv index + 1).
    pub(crate) fn total_count(&self) -> usize {
        let max_bv = self
            .principal_entries
            .iter()
            .map(|(bv, _)| *bv)
            .chain(self.projection_entries.iter().map(|(bv, _)| *bv))
            .max();
        max_bv.map_or(0, |m| m + 1)
    }
}

/// Location of a binder variable within the principal `ExistentialTraitRef`.
pub(crate) struct PrincipalLocation {
    /// Index of the arg within `ExistentialTraitRef.args` (excluding
    /// Self) in which this bv was found.
    pub(crate) arg_index: usize,
    /// TypeVisitor DFS offset of this bv *within* the arg at
    /// `arg_index`. A top-level lifetime arg has `dfs_offset == 0`
    /// and is the only region in that arg; a region nested inside a
    /// type arg (e.g., `&'b u8` when the arg is a type param) has
    /// a non-zero offset or shares the arg with other regions.
    pub(crate) dfs_offset: usize,
}

/// Location of a binder variable within an `ExistentialProjection`.
pub(crate) struct ProjectionLocation {
    /// `DefId` of the associated type.
    pub(crate) assoc_def_id: DefId,
    /// TypeVisitor DFS position of this bv within the projection's
    /// args (excluding Self) and term, walked in that order.
    pub(crate) dfs_position: usize,
}

// ── Functions ──────────────────────────────────────────────────────────────────

/// Walk a dyn type's existential predicates in TypeVisitor DFS order,
/// collecting all `ReBound` binder variables and their locations.
///
/// The canonical visitation order is: principal `ExistentialTraitRef`
/// (args excluding Self), then `ExistentialProjection`s sorted by def_id
/// (args excluding Self, then term). Binder variables are numbered by
/// order of first encounter.
pub(crate) fn collect_all_binder_vars<'tcx>(
    _tcx: TyCtxt<'tcx>,
    preds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> DynBinderVars {
    let mut result =
        DynBinderVars { principal_entries: Vec::new(), projection_entries: Vec::new() };

    let mut var_to_idx: FxHashMap<u32, usize> = FxHashMap::default();
    let mut next_bv_idx: usize = 0;
    // Counter for synthetic BoundVar indices assigned to ReErased
    // regions. Each erased position gets a unique value so that
    // intern_bv treats them as distinct binder variables.
    let mut next_erased_var: u32 = 0;

    // Assign a bv index on first encounter, return existing index
    // on subsequent encounters of the same BoundVar.
    let mut intern_bv = |var: ty::BoundVar| -> usize {
        *var_to_idx.entry(var.as_u32()).or_insert_with(|| {
            let idx = next_bv_idx;
            next_bv_idx += 1;
            idx
        })
    };

    // Principal trait ref.
    // ExistentialTraitRef.args already excludes Self, so no skip(1).
    if let Some(principal) = preds.principal() {
        for (arg_index, arg) in principal.skip_binder().args.iter().enumerate() {
            let regions = collect_bound_regions_in(arg, &mut next_erased_var);
            for (dfs_offset, br) in regions {
                let bv_idx = intern_bv(br.var);
                result
                    .principal_entries
                    .push((bv_idx, PrincipalLocation { arg_index, dfs_offset }));
            }
        }
    }

    // Projection predicates (sorted by def_id by construction).
    for proj_pred in preds.projection_bounds() {
        let proj = proj_pred.skip_binder();
        let assoc_def_id = proj.def_id;
        let mut dfs_position = 0;

        // Walk projection args. ExistentialProjection.args already
        // excludes Self (erased by `erase_self_ty`), so no skip(1).
        for arg in proj.args.iter() {
            let regions = collect_bound_regions_in(arg, &mut next_erased_var);
            for (offset, br) in regions {
                let bv_idx = intern_bv(br.var);
                result.projection_entries.push((
                    bv_idx,
                    ProjectionLocation { assoc_def_id, dfs_position: dfs_position + offset },
                ));
            }
            dfs_position += region_slots_of_arg(arg);
        }

        // Walk the projected term.
        let term_regions = collect_bound_regions_in(proj.term, &mut next_erased_var);
        for (offset, br) in term_regions {
            let bv_idx = intern_bv(br.var);
            result.projection_entries.push((
                bv_idx,
                ProjectionLocation { assoc_def_id, dfs_position: dfs_position + offset },
            ));
        }
    }

    result
}

/// Map from arg position (in `ExistentialTraitRef.args`, excluding
/// Self) to `ReEarlyParam.index` for lifetime-typed generic params.
/// Non-lifetime params are absent from the map.
///
/// The arg position `i` corresponds to
/// `generics_of(def_id).own_params[i+1]` (offset by 1 for Self). This
/// bridges between the arg-indexed `PrincipalLocation.arg_index` and the
/// `ReEarlyParam.index` used by where-clause derivation.
pub(crate) fn lifetime_param_map(tcx: TyCtxt<'_>, def_id: DefId) -> FxHashMap<usize, u32> {
    let generics = tcx.generics_of(def_id);
    generics
        .own_params
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.index != 0 // skip Self
                && matches!(p.kind, GenericParamDefKind::Lifetime)
        })
        .map(|(pos, p)| {
            // `pos` is the position in `own_params` (0-based,
            // includes Self at 0). Arg position in
            // ExistentialTraitRef.args is `pos - 1` (Self excluded).
            (pos - 1, p.index)
        })
        .collect()
}

/// Walk a `TypeVisitable` value with TypeVisitor DFS, collecting all
/// `ReBound` or `ReErased` regions and their DFS offset.
///
/// Returns `(dfs_offset, BoundRegion)` pairs in encounter order.
/// `dfs_offset` is the count of region-slots visited before this
/// region — a top-level lifetime arg yields a single `(0, br)` entry.
///
/// For `ReErased` regions (post-monomorphization), a synthetic
/// `BoundRegion` is created using `*next_erased_var` as the
/// `BoundVar` index, incremented for each erased region to ensure
/// uniqueness across calls.
pub(crate) fn collect_bound_regions_in<'tcx>(
    value: impl TypeVisitable<TyCtxt<'tcx>>,
    next_erased_var: &mut u32,
) -> Vec<(usize, ty::BoundRegion<'tcx>)> {
    struct Collector<'a, 'tcx> {
        regions: Vec<(usize, ty::BoundRegion<'tcx>)>,
        dfs_offset: usize,
        next_erased_var: &'a mut u32,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Collector<'_, 'tcx> {
        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            match r.kind() {
                ty::ReBound(_, br) => {
                    self.regions.push((self.dfs_offset, br));
                }
                ty::ReErased => {
                    let var = ty::BoundVar::from_u32(*self.next_erased_var);
                    *self.next_erased_var += 1;
                    let br = ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon };
                    self.regions.push((self.dfs_offset, br));
                }
                _ => {}
            }
            self.dfs_offset += 1;
        }
    }
    let mut collector = Collector { regions: Vec::new(), dfs_offset: 0, next_erased_var };
    value.visit_with(&mut collector);
    collector.regions
}

/// Count the total number of region slots in a `TypeVisitable` value
/// as visited by TypeVisitor DFS. This includes all regions regardless
/// of kind (`ReBound`, `ReEarlyParam`, `ReStatic`, etc.).
///
/// Used to compute `dfs_position` offsets when walking multiple args
/// in sequence (e.g., projection args followed by the projection term).
///
/// Generic fallback — prefer [`region_slots_of_arg`] /
/// [`region_slots_of_args`] / `Ty::region_slots()` when the input is a
/// `GenericArg`, `GenericArgsRef`, `Ty`, or `Const`, since those hit
/// an O(1) cache populated at interning.
pub(crate) fn count_region_slots_in<'tcx>(value: impl TypeVisitable<TyCtxt<'tcx>>) -> usize {
    struct Counter {
        count: usize,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Counter {
        fn visit_region(&mut self, _r: ty::Region<'tcx>) {
            self.count += 1;
        }
    }
    let mut counter = Counter { count: 0 };
    value.visit_with(&mut counter);
    counter.count
}

/// Region-slot count for a single `GenericArg`.
///
/// Dispatches on the kind tag and reads the cached count on the
/// interned `Ty` / `Const`; lifetime args contribute one slot,
/// `Outlives` args contribute none. O(1).
///
/// In debug builds the cached count is cross-checked against a live
/// `TypeVisitor` DFS. The check is retained in-tree permanently: it
/// costs nothing in release builds and catches any future `TyKind` /
/// `ConstKind` variant whose `FlagComputation` arm forgets to
/// propagate a child's region count.
#[inline]
pub(crate) fn region_slots_of_arg<'tcx>(arg: ty::GenericArg<'tcx>) -> usize {
    match arg.kind() {
        ty::GenericArgKind::Type(ty) => region_slots_of_ty(ty),
        ty::GenericArgKind::Lifetime(_) => 1,
        ty::GenericArgKind::Const(ct) => region_slots_of_const(ct),
        ty::GenericArgKind::Outlives(_) => 0,
    }
}

/// Region-slot count for an interned `&List<GenericArg>`.
///
/// Sums per-arg counts from the cache. O(n_args) with O(1) per arg.
#[inline]
pub(crate) fn region_slots_of_args<'tcx>(args: ty::GenericArgsRef<'tcx>) -> usize {
    args.iter().map(region_slots_of_arg).sum()
}

/// Region-slot count for a `Ty<'tcx>`. O(1); reads the cached value
/// stored on the interned `WithCachedTypeInfo<TyKind>`.
#[inline]
pub(crate) fn region_slots_of_ty<'tcx>(ty: Ty<'tcx>) -> usize {
    use rustc_middle::ty::Flags;
    let cached = ty.region_slots() as usize;
    debug_assert_eq!(
        cached,
        count_region_slots_in(ty),
        "cached Ty::region_slots disagrees with TypeVisitor walk (ty = {ty:?})",
    );
    cached
}

/// Region-slot count for a `Const<'tcx>`. O(1).
#[inline]
pub(crate) fn region_slots_of_const<'tcx>(ct: ty::Const<'tcx>) -> usize {
    use rustc_middle::ty::Flags;
    let cached = ct.region_slots() as usize;
    debug_assert_eq!(
        cached,
        count_region_slots_in(ct),
        "cached Const::region_slots disagrees with TypeVisitor walk (ct = {ct:?})",
    );
    cached
}

/// Region-slot count for a `Term<'tcx>` (either a `Ty` or a `Const`).
/// O(1) in both branches.
#[inline]
pub(crate) fn region_slots_of_term<'tcx>(term: ty::Term<'tcx>) -> usize {
    match term.kind() {
        ty::TermKind::Ty(ty) => region_slots_of_ty(ty),
        ty::TermKind::Const(ct) => region_slots_of_const(ct),
    }
}

/// Build a dense walk-position -> binder-variable mapping for a dyn
/// type's existential predicates.
///
/// The returned vector is indexed by the dyn type's raw DFS walk
/// position. Slots that are not binder-bearing are `None`; binder
/// occurrences map to their dense binder-variable index in
/// `collect_all_binder_vars` order.
fn dyn_walk_pos_to_bv_map<'tcx>(
    preds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<Option<usize>> {
    let mut map: Vec<Option<usize>> = Vec::new();

    let mut var_to_idx: FxHashMap<u32, usize> = FxHashMap::default();
    let mut next_bv_idx: usize = 0;
    let mut next_erased_var: u32 = 0;

    let mut intern_bv = |var: ty::BoundVar| -> usize {
        *var_to_idx.entry(var.as_u32()).or_insert_with(|| {
            let idx = next_bv_idx;
            next_bv_idx += 1;
            idx
        })
    };

    fn record_arg<'tcx, F: FnMut(ty::BoundVar) -> usize>(
        map: &mut Vec<Option<usize>>,
        arg: ty::GenericArg<'tcx>,
        walk_pos: &mut usize,
        next_erased_var: &mut u32,
        intern_bv: &mut F,
    ) {
        let region_slots = region_slots_of_arg(arg);
        if map.len() < *walk_pos + region_slots {
            map.resize(*walk_pos + region_slots, None);
        }
        for (dfs_offset, br) in collect_bound_regions_in(arg, next_erased_var) {
            let bv_idx = intern_bv(br.var);
            map[*walk_pos + dfs_offset] = Some(bv_idx);
        }
        *walk_pos += region_slots;
    }

    let mut walk_pos = 0usize;

    if let Some(principal) = preds.principal() {
        for arg in principal.skip_binder().args.iter() {
            record_arg(&mut map, arg, &mut walk_pos, &mut next_erased_var, &mut intern_bv);
        }
    }

    for proj_pred in preds.projection_bounds() {
        let proj = proj_pred.skip_binder();
        for arg in proj.args.iter() {
            record_arg(&mut map, arg, &mut walk_pos, &mut next_erased_var, &mut intern_bv);
        }

        let term_slots = region_slots_of_term(proj.term);
        if map.len() < walk_pos + term_slots {
            map.resize(walk_pos + term_slots, None);
        }
        for (dfs_offset, br) in collect_bound_regions_in(proj.term, &mut next_erased_var) {
            let bv_idx = intern_bv(br.var);
            map[walk_pos + dfs_offset] = Some(bv_idx);
        }
        walk_pos += term_slots;
    }

    map
}

struct TransportSegment<'a> {
    transport_start: usize,
    transport_slots: usize,
    walk_pos_to_bv: &'a [Option<usize>],
    native_base: usize,
}

fn remap_transport_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    segments: &[TransportSegment<'_>],
    call_site_outlives: &'tcx [ty::GenericArg<'tcx>],
) -> &'tcx [ty::GenericArg<'tcx>] {
    if call_site_outlives.is_empty() {
        return call_site_outlives;
    }

    let remap_index = |idx: usize| -> Option<usize> {
        if idx == usize::MAX {
            return Some(usize::MAX);
        }

        let Some(segment) = segments.iter().find(|segment| {
            idx >= segment.transport_start
                && idx < segment.transport_start + segment.transport_slots
        }) else {
            return None;
        };

        let local_idx = idx - segment.transport_start;
        segment.walk_pos_to_bv.get(local_idx).copied().flatten().map(|bv| segment.native_base + bv)
    };

    let remapped = call_site_outlives
        .iter()
        .filter_map(|entry| match entry.kind() {
            ty::GenericArgKind::Outlives(o) => {
                let longer = remap_index(o.longer())?;
                let shorter = remap_index(o.shorter())?;
                Some(tcx.mk_outlives_arg(longer, shorter).into())
            }
            _ => bug!("expected Outlives entry in call-site outlives slice"),
        })
        .collect::<Vec<_>>();

    tcx.arena.alloc_from_iter(remapped)
}

fn build_origin_to_native_map(
    segments: &[TransportSegment<'_>],
    origin_positions: &[Option<usize>],
) -> FxHashMap<usize, Vec<usize>> {
    let mut origin_to_native: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

    for segment in segments {
        for local_idx in 0..segment.transport_slots {
            let intrinsic_pos = segment.transport_start + local_idx;
            let Some(origin_pos) = origin_positions.get(intrinsic_pos).copied().flatten() else {
                continue;
            };
            let Some(native_bv) = segment.walk_pos_to_bv.get(local_idx).copied().flatten() else {
                continue;
            };
            origin_to_native.entry(origin_pos).or_default().push(segment.native_base + native_bv);
        }
    }

    origin_to_native
}

fn remap_origin_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    origin_to_native: &FxHashMap<usize, Vec<usize>>,
    call_site_outlives: &'tcx [ty::GenericArg<'tcx>],
) -> &'tcx [ty::GenericArg<'tcx>] {
    if call_site_outlives.is_empty() {
        return call_site_outlives;
    }

    let static_indices = [usize::MAX];
    let mut remapped = Vec::new();

    for entry in call_site_outlives {
        let ty::GenericArgKind::Outlives(outlives) = entry.kind() else {
            bug!("expected Outlives entry in call-site outlives slice");
        };

        let longer_indices = if outlives.longer() == usize::MAX {
            &static_indices[..]
        } else {
            let Some(indices) = origin_to_native.get(&outlives.longer()) else {
                continue;
            };
            indices.as_slice()
        };

        let shorter_indices = if outlives.shorter() == usize::MAX {
            &static_indices[..]
        } else {
            let Some(indices) = origin_to_native.get(&outlives.shorter()) else {
                continue;
            };
            indices.as_slice()
        };

        for &longer in longer_indices {
            for &shorter in shorter_indices {
                remapped.push((longer, shorter));
            }
        }
    }

    remapped.sort_unstable();
    remapped.dedup();

    tcx.arena.alloc_from_iter(
        remapped.into_iter().map(|(longer, shorter)| tcx.mk_outlives_arg(longer, shorter).into()),
    )
}

/// Remap transported outlives entries into the sub-trait binder-variable
/// space expected by `trait_metadata_index`.
pub(crate) fn remap_trait_metadata_outlives_entries<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_trait: Ty<'tcx>,
    sub_trait: Ty<'tcx>,
    call_site_outlives: &'tcx [ty::GenericArg<'tcx>],
) -> &'tcx [ty::GenericArg<'tcx>] {
    let (_super_data, sub_data) = match (*super_trait.kind(), *sub_trait.kind()) {
        (ty::Dynamic(s, ..), ty::Dynamic(t, ..)) => (s, t),
        _ => return call_site_outlives,
    };

    let sub_map = dyn_walk_pos_to_bv_map(sub_data);
    let root_transport_slots = region_slots_of_ty(super_trait);
    let sub_transport_slots = region_slots_of_ty(sub_trait);
    let root_drop_map: [Option<usize>; 0] = [];

    remap_transport_entries(
        tcx,
        &[
            TransportSegment {
                transport_start: 0,
                transport_slots: root_transport_slots,
                walk_pos_to_bv: &root_drop_map,
                native_base: 0,
            },
            TransportSegment {
                transport_start: root_transport_slots,
                transport_slots: sub_transport_slots,
                walk_pos_to_bv: &sub_map,
                native_base: 0,
            },
        ],
        call_site_outlives,
    )
}

/// Remap origin-space transported outlives entries into the sub-trait
/// binder-variable space expected by `trait_metadata_index`.
pub(crate) fn remap_trait_metadata_outlives_entries_from_origin_positions<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_trait: Ty<'tcx>,
    sub_trait: Ty<'tcx>,
    origin_positions: &[Option<usize>],
    call_site_outlives: &'tcx [ty::GenericArg<'tcx>],
) -> &'tcx [ty::GenericArg<'tcx>] {
    let (_super_data, sub_data) = match (*super_trait.kind(), *sub_trait.kind()) {
        (ty::Dynamic(s, ..), ty::Dynamic(t, ..)) => (s, t),
        _ => return call_site_outlives,
    };

    let sub_map = dyn_walk_pos_to_bv_map(sub_data);
    let root_transport_slots = region_slots_of_ty(super_trait);
    let sub_transport_slots = region_slots_of_ty(sub_trait);
    let root_drop_map: [Option<usize>; 0] = [];

    let origin_to_native = build_origin_to_native_map(
        &[
            TransportSegment {
                transport_start: 0,
                transport_slots: root_transport_slots,
                walk_pos_to_bv: &root_drop_map,
                native_base: 0,
            },
            TransportSegment {
                transport_start: root_transport_slots,
                transport_slots: sub_transport_slots,
                walk_pos_to_bv: &sub_map,
                native_base: 0,
            },
        ],
        origin_positions,
    );

    remap_origin_entries(tcx, &origin_to_native, call_site_outlives)
}

/// Compute the `trait_metadata_index` outlives class in the sub-trait's
/// native binder-variable space from an augmented intrinsic instance.
pub(crate) fn trait_metadata_index_outlives_class<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_trait: Ty<'tcx>,
    sub_trait: Ty<'tcx>,
    instance: ty::Instance<'tcx>,
) -> OutlivesClass<'tcx> {
    let transported = match instance.outlives_entries().split_first() {
        Some((_sentinel, entries)) => entries,
        None => &[],
    };
    let remapped = remap_trait_metadata_outlives_entries(tcx, super_trait, sub_trait, transported);
    OutlivesClass::from_entries(remapped)
}

/// Compute the set of target binder variables that are structurally exposed
/// through the root supertrait.
///
/// These target bvs participate in the root<->target correspondence checked
/// by `trait_cast_is_lifetime_erasure_safe`, so table admissibility does not
/// need to conservatively treat them like hidden Self-anchored lifetimes.
pub(crate) fn root_exposed_target_bvs<'tcx>(
    tcx: TyCtxt<'tcx>,
    root_trait: Ty<'tcx>,
    target_trait: Ty<'tcx>,
) -> DenseBitSet<usize> {
    let (root_data, target_data) = match (*root_trait.kind(), *target_trait.kind()) {
        (ty::Dynamic(root_data, ..), ty::Dynamic(target_data, ..)) => (root_data, target_data),
        _ => return DenseBitSet::new_empty(0),
    };

    let root_bvs = collect_all_binder_vars(tcx, root_data);
    let target_bvs = collect_all_binder_vars(tcx, target_data);
    let n_target = target_bvs.total_count();
    let mut exposed = DenseBitSet::new_empty(n_target);

    let (Some(root_principal), Some(target_principal)) =
        (root_data.principal(), target_data.principal())
    else {
        return exposed;
    };

    let root_def_id = root_principal.skip_binder().def_id;
    let target_def_id = target_principal.skip_binder().def_id;

    let root_principal_ref = root_principal.skip_binder();
    let target_principal_ref = target_principal.skip_binder();

    if root_def_id == target_def_id {
        // Same trait — identity correspondence. Walk root and target
        // args in parallel DFS; at each position where both sides have
        // a bv (ReErased or ReBound), mark the target bv as exposed.
        for (arg_index, (target_arg, root_arg)) in
            target_principal_ref.args.iter().zip(root_principal_ref.args.iter()).enumerate()
        {
            let target_regions = collect_all_regions_dfs(target_arg);
            let root_regions = collect_all_regions_dfs(root_arg);
            for (dfs_offset, (tr, rr)) in target_regions.iter().zip(root_regions.iter()).enumerate()
            {
                let is_target_bv = matches!(tr.kind(), ty::ReBound(..) | ty::ReErased);
                let is_root_bv = matches!(rr.kind(), ty::ReBound(..) | ty::ReErased);
                if is_target_bv && is_root_bv {
                    // Find the target bv index at this (arg_index, dfs_offset).
                    if let Some(&(bv_idx, _)) = target_bvs
                        .principal_entries
                        .iter()
                        .find(|(_, loc)| loc.arg_index == arg_index && loc.dfs_offset == dfs_offset)
                    {
                        exposed.insert(bv_idx);
                    }
                }
            }
        }
    } else {
        // Different traits — use synthetic ReBound propagation through
        // the supertrait chain, matching the technique used by
        // `compute_walk_pos_correspondences`.

        // Inject synthetic ReBound into target's erased regions.
        let mut next_synthetic_bv: u32 = 0;
        let modified_target_args: Vec<ty::GenericArg<'tcx>> = target_principal_ref
            .args
            .iter()
            .map(|arg| {
                ty::fold_regions(tcx, arg, |r, _depth| match r.kind() {
                    ty::ReErased => {
                        let var = ty::BoundVar::from_u32(next_synthetic_bv);
                        next_synthetic_bv += 1;
                        let br = ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon };
                        ty::Region::new_bound(tcx, ty::INNERMOST, br)
                    }
                    _ => r,
                })
            })
            .collect();
        let num_synthetic_bvs = next_synthetic_bv as usize;

        // Build modified ExistentialTraitRef → TraitRef → PolyTraitRef.
        let modified_existential_ref = ty::ExistentialTraitRef::new_from_args(
            tcx,
            target_def_id,
            tcx.mk_args_from_iter(modified_target_args.iter().copied()),
        );
        let dummy_self = tcx.types.trait_object_dummy_self;
        let target_trait_ref = modified_existential_ref.with_self_ty(tcx, dummy_self);
        let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
            (0..num_synthetic_bvs)
                .map(|_| ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon)),
        );
        let target_poly_trait_ref = ty::Binder::bind_with_vars(target_trait_ref, bound_vars);

        // Instantiate supertrait chain from target to root.
        let implied_root_trait_ref =
            instantiate_supertrait_chain(tcx, target_poly_trait_ref, root_def_id);

        if let Some(implied_root_trait_ref) = implied_root_trait_ref {
            // implied args include Self at position 0; actual args exclude Self.
            let implied_root_args = &implied_root_trait_ref.args[1..];
            let actual_root_args = root_principal_ref.args;

            // Walk implied vs actual root args in parallel DFS.
            for (implied_arg, actual_arg) in implied_root_args.iter().zip(actual_root_args.iter()) {
                let implied_regions = collect_all_regions_dfs(*implied_arg);
                let actual_regions = collect_all_regions_dfs(actual_arg);
                for (implied_r, actual_r) in implied_regions.iter().zip(actual_regions.iter()) {
                    if let ty::ReBound(_, implied_br) = implied_r.kind()
                        && implied_br.var.as_usize() < num_synthetic_bvs
                        && matches!(actual_r.kind(), ty::ReErased | ty::ReBound(..))
                    {
                        // synthetic_var_i maps to target bv index i
                        // (both assigned in same DFS order over principal args).
                        exposed.insert(implied_br.var.as_usize());
                    }
                }
            }
        }
    }

    for &(target_bv_idx, ref proj_loc) in &target_bvs.projection_entries {
        if root_bvs.projection_entries.iter().any(|(_, loc)| {
            loc.assoc_def_id == proj_loc.assoc_def_id && loc.dfs_position == proj_loc.dfs_position
        }) {
            exposed.insert(target_bv_idx);
        }
    }

    exposed
}

// ── Walk-position-based erasure safety ─────────────────────────────────────────

/// Structural correspondences between target and root dyn types expressed
/// in predicate walk-position space. Each pair `(t_wp, r_wp)` means:
/// "the target lifetime at predicate walk position `t_wp` structurally
/// corresponds to the root lifetime at predicate walk position `r_wp`
/// through the supertrait chain."
struct WalkPosCorrespondences {
    /// `(target_pred_walk_pos, root_pred_walk_pos)` pairs.
    pairs: Vec<(usize, usize)>,
    /// `false` if surjectivity fails: some root lifetime param has no
    /// corresponding target param, or some target param reaches no root
    /// param.
    surjective: bool,
}

/// Compute structural correspondences between target and root dyn types
/// in predicate walk-position space.
///
/// Monomorphizes the supertrait chain from target → root using
/// `instantiate_supertrait`, then structurally compares the implied root
/// args with the actual root args, pairing up `ReBound` region positions.
/// This handles lifetimes nested inside type params (e.g.,
/// `trait Target<'a>: Root<&'a u8>`).
fn compute_walk_pos_correspondences<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    target_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> WalkPosCorrespondences {
    let (Some(super_principal), Some(target_principal)) =
        (super_data.principal(), target_data.principal())
    else {
        return WalkPosCorrespondences { pairs: Vec::new(), surjective: true };
    };

    let super_def_id = super_principal.skip_binder().def_id;
    let target_def_id = target_principal.skip_binder().def_id;

    if super_def_id == target_def_id {
        // Same trait — identity correspondence on all ReBound positions.
        return compute_identity_correspondences(tcx, super_data, target_data);
    }

    let super_principal_ref = super_principal.skip_binder();
    let target_principal_ref = target_principal.skip_binder();

    // --- Inject synthetic ReBound identities into the target args -----
    //
    // Post-monomorphization all regions are ReErased, so the downstream
    // `instantiate_supertrait_chain` and matching loop see nothing useful.
    // We replace each ReErased in the target's existential args with
    // `ReBound(INNERMOST, BV_i)`, giving every erased region slot a
    // unique identity that survives substitution through the supertrait
    // chain.
    let mut next_synthetic_bv: u32 = 0;
    let modified_target_args: Vec<ty::GenericArg<'tcx>> = target_principal_ref
        .args
        .iter()
        .map(|arg| {
            ty::fold_regions(tcx, arg, |r, _depth| match r.kind() {
                ty::ReErased => {
                    let var = ty::BoundVar::from_u32(next_synthetic_bv);
                    next_synthetic_bv += 1;
                    let br = ty::BoundRegion { var, kind: ty::BoundRegionKind::Anon };
                    ty::Region::new_bound(tcx, ty::INNERMOST, br)
                }
                _ => r,
            })
        })
        .collect();
    let num_synthetic_bvs = next_synthetic_bv as usize;

    // Build a modified ExistentialTraitRef with synthetic ReBound args.
    let modified_existential_ref = ty::ExistentialTraitRef::new_from_args(
        tcx,
        target_def_id,
        tcx.mk_args_from_iter(modified_target_args.iter().copied()),
    );

    // Convert to TraitRef (needs a Self type; we use a dummy since Self
    // is erased in dyn types).
    let dummy_self = tcx.types.trait_object_dummy_self;
    let target_trait_ref = modified_existential_ref.with_self_ty(tcx, dummy_self);

    // Build a PolyTraitRef with N bound variable slots for the synthetic BVs.
    let bound_vars = tcx.mk_bound_variable_kinds_from_iter(
        (0..num_synthetic_bvs).map(|_| ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon)),
    );
    let target_poly_trait_ref = ty::Binder::bind_with_vars(target_trait_ref, bound_vars);

    // Walk the supertrait chain from target_def_id toward super_def_id,
    // instantiating at each step. This is the same pattern as
    // `prepare_vtable_segments` in vtable.rs.
    let implied_root_trait_ref =
        instantiate_supertrait_chain(tcx, target_poly_trait_ref, super_def_id);

    let Some(implied_root_trait_ref) = implied_root_trait_ref else {
        // No path from target to root through the supertrait chain.
        return WalkPosCorrespondences { pairs: Vec::new(), surjective: false };
    };

    // The implied root args include Self at position 0; the existential
    // args exclude Self. Skip the first arg (Self) when comparing.
    let implied_root_args = &implied_root_trait_ref.args[1..];
    let actual_root_args = super_principal_ref.args;

    // --- Structural comparison of implied vs actual root args ---------
    //
    // Walk both arg lists in parallel DFS. At each `ReBound` region in
    // the implied args (with var index < num_synthetic_bvs), record the
    // target walk position; at the same DFS position in the actual root
    // args, record the root walk position.
    let mut pairs: Vec<(usize, usize)> = Vec::new();

    // Walk the MODIFIED target args (now ReBound) to build a map from
    // BoundVar → walk position in the target predicate space.
    let target_bound_wp = collect_rebound_walk_positions(&modified_target_args);

    let mut root_wp_offset = 0usize;
    for (implied_arg, actual_arg) in implied_root_args.iter().zip(actual_root_args.iter()) {
        let implied_regions = collect_all_regions_dfs(*implied_arg);
        let actual_regions = collect_all_regions_dfs(actual_arg);

        for (dfs_offset, (implied_r, _actual_r)) in
            implied_regions.iter().zip(actual_regions.iter()).enumerate()
        {
            // Only pair up positions where the implied side has a
            // synthetic ReBound region (var < num_synthetic_bvs). The
            // actual side may be ReErased — that's fine because we
            // identify its position purely by DFS offset.
            if let ty::ReBound(_, implied_br) = implied_r.kind()
                && implied_br.var.as_usize() < num_synthetic_bvs
            {
                // Find the target walk position for this binder var.
                if let Some(&target_wp) = target_bound_wp.get(&implied_br.var) {
                    let root_wp = root_wp_offset + dfs_offset;
                    pairs.push((target_wp, root_wp));
                }
            }
        }

        root_wp_offset += region_slots_of_arg(actual_arg);
    }

    // --- Projection predicates ----------------------------------------
    let super_proj_base: usize = region_slots_of_args(super_principal_ref.args);
    let target_proj_base: usize = region_slots_of_args(target_principal_ref.args);

    let super_projs: Vec<_> = super_data.projection_bounds().collect();
    let target_projs: Vec<_> = target_data.projection_bounds().collect();

    // Compute cumulative walk-position offsets for each super projection.
    let mut super_proj_offsets: FxHashMap<DefId, usize> = FxHashMap::default();
    {
        let mut offset = super_proj_base;
        for proj_pred in &super_projs {
            let proj = proj_pred.skip_binder();
            super_proj_offsets.insert(proj.def_id, offset);
            let arg_slots: usize = region_slots_of_args(proj.args);
            let term_slots = region_slots_of_term(proj.term);
            offset += arg_slots + term_slots;
        }
    }

    // Walk target projections and match against root by assoc_def_id.
    {
        let mut target_offset = target_proj_base;
        for target_proj_pred in &target_projs {
            let target_proj = target_proj_pred.skip_binder();
            let target_assoc = target_proj.def_id;
            let target_arg_slots: usize = region_slots_of_args(target_proj.args);
            let target_term_slots = region_slots_of_term(target_proj.term);
            let target_total = target_arg_slots + target_term_slots;

            if let Some(&super_offset) = super_proj_offsets.get(&target_assoc) {
                let super_proj_pred =
                    super_projs.iter().find(|p| p.skip_binder().def_id == target_assoc).unwrap();
                let super_proj = super_proj_pred.skip_binder();

                let mut erased_counter = u32::MAX / 2;
                let target_regions =
                    collect_projection_bound_positions(target_proj, &mut erased_counter);
                let super_regions =
                    collect_projection_bound_positions(super_proj, &mut erased_counter);

                for &t_pos in &target_regions {
                    if super_regions.contains(&t_pos) {
                        pairs.push((target_offset + t_pos, super_offset + t_pos));
                    }
                }
            }

            target_offset += target_total;
        }
    }

    // --- Surjectivity -------------------------------------------------
    //
    // Count total ReBound regions in each dyn type's predicates. Every
    // root ReBound must be reached by some target ReBound, and vice
    // versa. We check this by counting distinct positions in the pairs.
    let total_target_rebound = count_rebound_regions_in_preds(target_data);
    let total_root_rebound = count_rebound_regions_in_preds(super_data);

    let target_wps_in_pairs: FxHashSet<usize> = pairs.iter().map(|&(t, _)| t).collect();
    let root_wps_in_pairs: FxHashSet<usize> = pairs.iter().map(|&(_, r)| r).collect();

    let surjective = target_wps_in_pairs.len() >= total_target_rebound
        && root_wps_in_pairs.len() >= total_root_rebound;

    WalkPosCorrespondences { pairs, surjective }
}

/// Identity correspondences when root and target are the same trait.
///
/// For the identity case, every region slot at the same position in both
/// sides corresponds to itself: `(wp_i, wp_i)` for each region slot.
/// This handles both `ReBound` (pre-mono) and `ReErased` (post-mono)
/// regions.
fn compute_identity_correspondences<'tcx>(
    _tcx: TyCtxt<'tcx>,
    super_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    target_data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> WalkPosCorrespondences {
    let super_principal = super_data.principal().unwrap();
    let target_principal = target_data.principal().unwrap();
    let super_ref = super_principal.skip_binder();
    let target_ref = target_principal.skip_binder();

    let mut pairs = Vec::new();
    let mut target_wp = 0usize;
    let mut root_wp = 0usize;

    for (target_arg, root_arg) in target_ref.args.iter().zip(super_ref.args.iter()) {
        let target_regions = collect_all_regions_dfs(target_arg);
        let root_regions = collect_all_regions_dfs(root_arg);

        for (offset, (tr, rr)) in target_regions.iter().zip(root_regions.iter()).enumerate() {
            // Accept both ReBound (pre-mono) and ReErased (post-mono)
            // regions. In the identity case, every region slot
            // corresponds to itself.
            let is_target_region = matches!(tr.kind(), ty::ReBound(..) | ty::ReErased);
            let is_root_region = matches!(rr.kind(), ty::ReBound(..) | ty::ReErased);
            if is_target_region && is_root_region {
                pairs.push((target_wp + offset, root_wp + offset));
            }
        }

        target_wp += region_slots_of_arg(target_arg);
        root_wp += region_slots_of_arg(root_arg);
    }

    WalkPosCorrespondences { pairs, surjective: true }
}

/// Walk the supertrait chain from `start` toward `target_def_id` using
/// `instantiate_supertrait`. Returns the monomorphized `TraitRef` at
/// `target_def_id`, or `None` if no path exists.
fn instantiate_supertrait_chain<'tcx>(
    tcx: TyCtxt<'tcx>,
    start: ty::PolyTraitRef<'tcx>,
    target_def_id: DefId,
) -> Option<ty::TraitRef<'tcx>> {
    // BFS through the supertrait hierarchy.
    let mut queue = std::collections::VecDeque::new();
    let mut visited = FxHashSet::default();
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        let current_def_id = current.def_id();
        if !visited.insert(current_def_id) {
            continue;
        }

        if current_def_id == target_def_id {
            return Some(current.skip_binder());
        }

        let super_predicates = tcx.explicit_super_predicates_of(current_def_id);
        for (pred, _) in
            super_predicates.iter_identity_copied().map(ty::Unnormalized::skip_norm_wip)
        {
            let Some(trait_clause) = pred.instantiate_supertrait(tcx, current).as_trait_clause()
            else {
                continue;
            };
            let parent_ref = trait_clause.map_bound(|tc| tc.trait_ref);
            queue.push_back(parent_ref);
        }
    }

    None
}

/// Collect all regions encountered during TypeVisitor DFS of a GenericArg.
fn collect_all_regions_dfs<'tcx>(arg: ty::GenericArg<'tcx>) -> Vec<ty::Region<'tcx>> {
    struct Collector<'tcx> {
        regions: Vec<ty::Region<'tcx>>,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Collector<'tcx> {
        fn visit_region(&mut self, r: ty::Region<'tcx>) {
            self.regions.push(r);
        }
    }
    let mut collector = Collector { regions: Vec::new() };
    arg.visit_with(&mut collector);
    collector.regions
}

/// Build a map from `ReBound` `BoundVar` → predicate walk position for
/// a principal `ExistentialTraitRef`'s args.
fn collect_rebound_walk_positions(args: &[ty::GenericArg<'_>]) -> FxHashMap<ty::BoundVar, usize> {
    let mut map = FxHashMap::default();
    let mut wp = 0usize;
    for arg in args.iter() {
        let regions = collect_all_regions_dfs(*arg);
        for (offset, r) in regions.iter().enumerate() {
            if let ty::ReBound(_, br) = r.kind() {
                // First occurrence wins (a bv may appear in multiple
                // args; the walk position is its first DFS hit).
                map.entry(br.var).or_insert(wp + offset);
            }
        }
        wp += region_slots_of_arg(*arg);
    }
    map
}

/// Count the total number of `ReBound` or `ReErased` region positions
/// in a dyn type's existential predicates (principal + projections).
///
/// Post-monomorphization all regions are `ReErased`, so both kinds must
/// be counted to ensure the surjectivity check is meaningful.
fn count_rebound_regions_in_preds<'tcx>(
    preds: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> usize {
    let mut count = 0usize;
    if let Some(principal) = preds.principal() {
        for arg in principal.skip_binder().args.iter() {
            for r in collect_all_regions_dfs(arg) {
                if matches!(r.kind(), ty::ReBound(..) | ty::ReErased) {
                    count += 1;
                }
            }
        }
    }
    for proj_pred in preds.projection_bounds() {
        let proj = proj_pred.skip_binder();
        for arg in proj.args.iter() {
            for r in collect_all_regions_dfs(arg) {
                if matches!(r.kind(), ty::ReBound(..) | ty::ReErased) {
                    count += 1;
                }
            }
        }
        // Term regions.
        struct Counter {
            count: usize,
        }
        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Counter {
            fn visit_region(&mut self, r: ty::Region<'tcx>) {
                if matches!(r.kind(), ty::ReBound(..) | ty::ReErased) {
                    self.count += 1;
                }
            }
        }
        let mut counter = Counter { count: 0 };
        proj.term.visit_with(&mut counter);
        count += counter.count;
    }
    count
}

/// Collect the DFS positions of `ReBound` or `ReErased` regions within
/// an `ExistentialProjection`'s args + term. Positions are relative to
/// the start of the projection (not the dyn type).
///
/// Both `ReBound` and `ReErased` regions are included so that the
/// projection correspondence check works post-monomorphization.
fn collect_projection_bound_positions<'tcx>(
    proj: ty::ExistentialProjection<'tcx>,
    next_erased_var: &mut u32,
) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut walk_pos = 0usize;

    for arg in proj.args.iter() {
        let regions = collect_bound_regions_in(arg, next_erased_var);
        for (offset, _br) in &regions {
            positions.push(walk_pos + offset);
        }
        walk_pos += region_slots_of_arg(arg);
    }

    let term_regions = collect_bound_regions_in(proj.term, next_erased_var);
    for (offset, _br) in &term_regions {
        positions.push(walk_pos + offset);
    }

    positions
}

// ── Resolution ─────────────────────────────────────────────────────────────────

/// Resolve `trait_cast_is_lifetime_erasure_safe` in walk-position space.
///
/// Returns `Ok(())` iff:
///   1. The structural walk-position correspondences (from
///      `compute_walk_pos_correspondences`) are surjective.
///   2. Every (target_wp, root_wp) pair has mutual outlives in the
///      caller's environment when translated through `origin_positions`.
///
/// Returns `Err(reason)` with a short static string describing which
/// admissibility rule rejected the cast; consumed by the
/// `-Zdump-trait-cast-erasure-safety` diagnostic.
fn resolve_erasure_safe_walk_pos<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_trait: Ty<'tcx>,
    target_trait: Ty<'tcx>,
    origin_positions: &[Option<usize>],
    call_site_outlives: &[ty::GenericArg<'tcx>],
    root_transport_slots: usize,
) -> Result<(), &'static str> {
    let (super_data, target_data) = match (*super_trait.kind(), *target_trait.kind()) {
        (ty::Dynamic(s, ..), ty::Dynamic(t, ..)) => (s, t),
        _ => return Ok(()),
    };

    let correspondences = compute_walk_pos_correspondences(tcx, super_data, target_data);

    if !correspondences.surjective {
        return Err("non-surjective correspondences");
    }

    if correspondences.pairs.is_empty() {
        return Ok(());
    }

    // Build an outlives oracle from the call-site entries (already in
    // origin walk-position space). Compute the dimension (max index + 2
    // for 0-based + 'static slot) and use the cached Floyd-Warshall
    // reachability matrix for O(1) outlives lookups.
    let max_idx = call_site_outlives
        .iter()
        .filter_map(|entry| match entry.kind() {
            ty::GenericArgKind::Outlives(o) => {
                let l = if o.longer() == usize::MAX { None } else { Some(o.longer()) };
                let s = if o.shorter() == usize::MAX { None } else { Some(o.shorter()) };
                l.into_iter().chain(s).max()
            }
            _ => bug!("expected Outlives entry in call-site outlives slice"),
        })
        .max()
        .unwrap_or(0);
    let dim = max_idx + 2;
    let interned_entries = tcx.arena.alloc_from_iter(call_site_outlives.iter().copied());
    let reach = tcx.outlives_reachability((interned_entries, dim));
    let env = crate::cast_sensitivity::CallerOutlivesEnv::from_raw(reach, dim);

    // Check mutual outlives for each structural pair.
    // Group pairs by target walk position to handle fan-out.
    let mut target_to_roots: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for &(t_wp, r_wp) in &correspondences.pairs {
        // Translate predicate walk positions to transport positions,
        // then to origin positions.
        let t_transport = root_transport_slots + t_wp;
        let r_transport = r_wp;

        let Some(origin_t) = origin_positions.get(t_transport).copied().flatten() else {
            return Err("target origin position missing");
        };
        let Some(origin_r) = origin_positions.get(r_transport).copied().flatten() else {
            return Err("root origin position missing");
        };

        // Mutual outlives: target and root represent the same lifetime.
        if !env.outlives(origin_t, origin_r) || !env.outlives(origin_r, origin_t) {
            return Err("mutual-outlives missing for structural pair");
        }

        target_to_roots.entry(t_wp).or_default().push(origin_r);
    }

    // Fan-out check: when one target maps to multiple roots, those
    // roots must also be mutually equivalent.
    // Iteration order is irrelevant — we check a universal property.
    #[allow(rustc::potential_query_instability)]
    for roots in target_to_roots.values() {
        for i in 0..roots.len() {
            for j in (i + 1)..roots.len() {
                if !env.outlives(roots[i], roots[j]) || !env.outlives(roots[j], roots[i]) {
                    return Err("fan-out mutual-outlives missing");
                }
            }
        }
    }

    Ok(())
}

/// Emit the `-Zdump-trait-cast-erasure-safety` diagnostic block for a
/// single erasure-safety query.
///
/// Only invoked when the flag is set and the super-trait's printed name
/// matches the filter. This function recomputes the auxiliary data
/// (binder-var enumeration, supertrait chain) from scratch — it is a
/// strict observer of analysis inputs and the already-computed verdict.
fn dump_erasure_safety<'tcx>(
    tcx: TyCtxt<'tcx>,
    super_trait: Ty<'tcx>,
    target_trait: Ty<'tcx>,
    call_site_outlives: &[ty::GenericArg<'tcx>],
    verdict: Result<(), &'static str>,
) {
    let super_name = with_no_trimmed_paths!(super_trait.to_string());
    let target_name = with_no_trimmed_paths!(target_trait.to_string());

    eprintln!("=== Erasure Safety: super={super_name} target={target_name} ===");

    // Outlives class: print the semantic (longer, shorter) pairs.
    if call_site_outlives.is_empty() {
        eprintln!("  Outlives class: empty");
    } else {
        let mut pairs: Vec<(usize, usize)> = call_site_outlives
            .iter()
            .filter_map(|entry| match entry.kind() {
                ty::GenericArgKind::Outlives(o) => Some((o.longer(), o.shorter())),
                _ => None,
            })
            .collect();
        pairs.sort_unstable();
        let rendered: Vec<String> = pairs
            .iter()
            .map(|&(l, s)| {
                let l_s = if l == usize::MAX { "'static".to_string() } else { format!("wp{l}") };
                let s_s = if s == usize::MAX { "'static".to_string() } else { format!("wp{s}") };
                format!("{l_s}: {s_s}")
            })
            .collect();
        eprintln!("  Outlives class: [{}]", rendered.join(", "));
    }

    // Binder var enumeration for the super-trait and target-trait.
    let super_data = match *super_trait.kind() {
        ty::Dynamic(d, ..) => Some(d),
        _ => None,
    };
    let target_data = match *target_trait.kind() {
        ty::Dynamic(d, ..) => Some(d),
        _ => None,
    };

    let (mut principal_entries, mut projection_entries) = target_data
        .map(|d| {
            let bvs = collect_all_binder_vars(tcx, d);
            (bvs.principal_entries, bvs.projection_entries)
        })
        .unwrap_or_else(|| (Vec::new(), Vec::new()));
    // `collect_all_binder_vars` yields entries in DFS order; sort stably
    // by (bv_index, arg_index / assoc, dfs_offset) for deterministic
    // output independent of any upstream hash iteration.
    principal_entries.sort_by_key(|(bv, loc)| (*bv, loc.arg_index, loc.dfs_offset));
    projection_entries
        .sort_by_key(|(bv, loc)| (*bv, loc.assoc_def_id.index.as_u32(), loc.dfs_position));

    eprintln!("  Principal binder vars ({}):", principal_entries.len());
    for (bv, loc) in &principal_entries {
        eprintln!("    bv{bv}: arg_index={} dfs_offset={}", loc.arg_index, loc.dfs_offset);
    }
    eprintln!("  Projection binder vars ({}):", projection_entries.len());
    for (bv, loc) in &projection_entries {
        let assoc = with_no_trimmed_paths!(tcx.def_path_str(loc.assoc_def_id));
        eprintln!("    bv{bv}: assoc={assoc} dfs_position={}", loc.dfs_position);
    }

    // Where-clause / structural correspondences derivation summary.
    if let (Some(sd), Some(td)) = (super_data, target_data) {
        let corr = compute_walk_pos_correspondences(tcx, sd, td);
        let mut pairs = corr.pairs.clone();
        pairs.sort_unstable();
        eprintln!(
            "  Where-clause outlives derivation: surjective={} pairs={}",
            corr.surjective,
            pairs.len()
        );
        for (t_wp, r_wp) in pairs {
            eprintln!("    target_wp={t_wp} <-> root_wp={r_wp}");
        }
    } else {
        eprintln!("  Where-clause outlives derivation: (non-dyn query — vacuously safe)");
    }

    // Supertrait chain trace from target principal → super principal.
    eprintln!("  Supertrait chain:");
    if let (Some(sd), Some(td)) = (super_data, target_data)
        && let (Some(sp), Some(tp)) = (sd.principal(), td.principal())
    {
        let super_def_id = sp.skip_binder().def_id;
        let target_def_id = tp.skip_binder().def_id;
        let chain = trace_supertrait_chain(tcx, target_def_id, super_def_id);
        match chain {
            Some(defs) => {
                let rendered: Vec<String> =
                    defs.iter().map(|d| with_no_trimmed_paths!(tcx.def_path_str(*d))).collect();
                eprintln!("    {}", rendered.join(" -> "));
            }
            None => eprintln!("    (no path from target to super in supertrait hierarchy)"),
        }
    } else {
        eprintln!("    (non-dyn query — no chain)");
    }

    match verdict {
        Ok(()) => eprintln!("  Verdict: safe"),
        Err(reason) => eprintln!("  Verdict: unsafe ({reason})"),
    }
}

/// Trace the shortest supertrait path from `start` to `target`.
///
/// Returns the list of `DefId`s along the path (`start` first, `target`
/// last), or `None` if no such path exists. Used only for the
/// `-Zdump-trait-cast-erasure-safety` diagnostic; the real analysis uses
/// `instantiate_supertrait_chain` which substitutes args.
fn trace_supertrait_chain(tcx: TyCtxt<'_>, start: DefId, target: DefId) -> Option<Vec<DefId>> {
    if start == target {
        return Some(vec![start]);
    }
    let mut queue: std::collections::VecDeque<(DefId, Vec<DefId>)> =
        std::collections::VecDeque::new();
    let mut visited: FxHashSet<DefId> = FxHashSet::default();
    queue.push_back((start, vec![start]));
    visited.insert(start);
    while let Some((cur, path)) = queue.pop_front() {
        let supers = tcx.explicit_super_predicates_of(cur);
        // Collect parent trait def_ids in a deterministic order.
        let mut parents: Vec<DefId> = supers
            .iter_identity_copied()
            .map(ty::Unnormalized::skip_norm_wip)
            .filter_map(|(pred, _)| {
                pred.as_trait_clause().map(|tc| tc.skip_binder().trait_ref.def_id)
            })
            .collect();
        parents.sort_by_key(|d| (d.krate.as_u32(), d.index.as_u32()));
        parents.dedup();
        for parent in parents {
            if !visited.insert(parent) {
                continue;
            }
            let mut next_path = path.clone();
            next_path.push(parent);
            if parent == target {
                return Some(next_path);
            }
            queue.push_back((parent, next_path));
        }
    }
    None
}

/// Query provider for `is_lifetime_erasure_safe`.
///
/// Determines whether casting to `target_trait` within the graph rooted at
/// `super_trait` is safe w.r.t. lifetime erasure, given call-site outlives
/// entries and origin walk positions from the CRL composition pipeline.
pub(crate) fn is_lifetime_erasure_safe<'tcx>(
    tcx: TyCtxt<'tcx>,
    (super_trait, target_trait, origin_positions, call_site_outlives): (
        Ty<'tcx>,
        Ty<'tcx>,
        &'tcx [Option<usize>],
        &'tcx [ty::GenericArg<'tcx>],
    ),
) -> bool {
    let root_transport_slots = region_slots_of_ty(super_trait);
    let verdict = resolve_erasure_safe_walk_pos(
        tcx,
        super_trait,
        target_trait,
        origin_positions,
        call_site_outlives,
        root_transport_slots,
    );

    // `-Zdump-trait-cast-erasure-safety` diagnostic emission. Fast path
    // when the flag is absent; substring match against the super-trait's
    // fully-qualified printed name when present.
    if let Some(filter) = tcx.sess.opts.unstable_opts.dump_trait_cast_erasure_safety.as_deref() {
        let super_name = with_no_trimmed_paths!(super_trait.to_string());
        let matches = filter == "all" || super_name.contains(filter);
        if matches {
            dump_erasure_safety(tcx, super_trait, target_trait, call_site_outlives, verdict);
        }
    }

    verdict.is_ok()
}
