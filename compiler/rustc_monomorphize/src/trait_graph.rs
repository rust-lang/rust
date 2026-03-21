use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir::def_id::DefId;
use rustc_index::bit_set::{BitMatrix, DenseBitSet};
use rustc_middle::bug;
use rustc_middle::traits::{CodegenObligationError, ImplSource, ImplSourceUserDefinedData};
use rustc_middle::ty::trait_cast::{FingerprintedTy, SubTraitInfo, TraitGraph};
use rustc_middle::ty::{
    self, EarlyParamRegion, GenericArg, RegionVid, Ty, TyCtxt, TypeVisitable, TypeVisitor,
};

use crate::erasure_safe::{
    collect_all_binder_vars, lifetime_param_map, trait_metadata_index_outlives_class,
};

/// Build a [`TraitGraph`] for a given root supertrait.
///
/// Partitions the gathered delayed-codegen requests into sub-trait →
/// outlives-class mappings and the set of concrete types that requested
/// metadata tables. Requests whose `super_trait` does not match `root`
/// are ignored (they belong to a different root's graph).
pub(crate) fn trait_cast_graph<'tcx>(tcx: TyCtxt<'tcx>, root: Ty<'tcx>) -> TraitGraph<'tcx> {
    let requests = tcx.gather_trait_cast_requests(());

    // Build with FxHash* for .entry() access, then wrap in UnordMap/UnordSet.
    let mut sub_traits: UnordMap<FingerprintedTy<'tcx>, SubTraitInfo<'tcx>> = UnordMap::default();
    let mut concrete_types: UnordSet<FingerprintedTy<'tcx>> = UnordSet::default();

    for req in &requests.index_requests {
        if req.super_trait != root {
            continue;
        }
        let outlives_class =
            trait_metadata_index_outlives_class(tcx, req.super_trait, req.sub_trait, req.instance);
        let key = FingerprintedTy::new(tcx, req.sub_trait);
        sub_traits
            .entry(key)
            .or_insert_with(|| SubTraitInfo { outlives_classes: UnordSet::default() })
            .outlives_classes
            .insert(outlives_class);
    }

    for req in &requests.table_requests {
        if req.super_trait != root {
            continue;
        }
        concrete_types.insert(FingerprintedTy::new(tcx, req.concrete_type));
    }

    // TableLenRequests don't contribute additional structure — the root
    // is already the key. Their presence in the requests ensures the root
    // appears in the `root_traits()` set that drives invocation, but the
    // graph itself needs no extra state from them.
    //
    // ErasureSafeRequests are resolved independently — they query the
    // layout but do not influence graph construction.

    // Ensure the table layout includes slots for where-clause-derived
    // outlives classes. Trait where-clauses like `where 'a: 'b` create
    // admissibility requirements that the empty outlives class cannot
    // satisfy. Without the corresponding slot, casts through generic
    // library code (e.g. `TraitCast::unchecked_cast`) that carry valid
    // outlives evidence would find only a null table entry.
    let wc_additions: Vec<_> = sub_traits
        .items()
        .map(|(k, v)| (*k, v))
        .into_sorted_stable_ord_by_key(|item| &item.0)
        .into_iter()
        .filter_map(|(key, info)| {
            if info.outlives_classes.is_empty() {
                return None;
            }
            let sub_trait: Ty<'tcx> = *key;
            derive_where_clause_outlives_class(tcx, sub_trait).map(|cls| (key, cls))
        })
        .collect();
    for (key, wc_class) in wc_additions {
        sub_traits.get_mut(&key).unwrap().outlives_classes.insert(wc_class);
    }

    TraitGraph { root, sub_traits, concrete_types }
}

/// Derive an outlives class from a sub-trait's where-clauses on lifetime
/// parameters.
///
/// For a sub-trait `dyn Trait<'a, 'b>` where the trait has `where 'a: 'b`,
/// this returns an `OutlivesClass` with `(bv0, bv1)` — the binder variable
/// pair corresponding to the where-clause's region outlives predicate.
///
/// Returns `None` if the trait has no region outlives predicates on its
/// own lifetime parameters, or if the dyn type has no principal trait.
#[allow(rustc::potential_query_instability)]
pub(crate) fn derive_where_clause_outlives_class<'tcx>(
    tcx: TyCtxt<'tcx>,
    sub_trait: Ty<'tcx>,
) -> Option<ty::trait_cast::OutlivesClass<'tcx>> {
    let ty::Dynamic(dyn_data, ..) = sub_trait.kind() else {
        return None;
    };
    let principal = dyn_data.principal()?;
    let trait_def_id = principal.skip_binder().def_id;

    // Map from arg position (in ExistentialTraitRef.args) to param index.
    let lt_map = lifetime_param_map(tcx, trait_def_id);
    // Invert: param_index → arg_position.
    let param_to_arg: FxHashMap<u32, usize> =
        lt_map.iter().map(|(&arg_pos, &param_idx)| (param_idx, arg_pos)).collect();

    // Map arg position to binder variable index.
    let bvs = collect_all_binder_vars(tcx, dyn_data);
    let mut arg_to_bv: FxHashMap<usize, usize> = FxHashMap::default();
    for &(bv_idx, ref loc) in &bvs.principal_entries {
        if loc.dfs_offset == 0 {
            arg_to_bv.insert(loc.arg_index, bv_idx);
        }
    }

    // Walk the trait's predicates for RegionOutlives clauses.
    let predicates = tcx.predicates_of(trait_def_id);
    let mut entries = Vec::new();

    for (pred, _) in predicates.predicates {
        if let ty::ClauseKind::RegionOutlives(outlives) = pred.kind().skip_binder() {
            let longer_param = match outlives.0.kind() {
                ty::ReEarlyParam(ep) => Some(ep.index),
                _ => None,
            };
            let shorter_param = match outlives.1.kind() {
                ty::ReEarlyParam(ep) => Some(ep.index),
                _ => None,
            };

            if let (Some(longer_param), Some(shorter_param)) = (longer_param, shorter_param) {
                let longer_arg = param_to_arg.get(&longer_param);
                let shorter_arg = param_to_arg.get(&shorter_param);
                if let (Some(&la), Some(&sa)) = (longer_arg, shorter_arg) {
                    let longer_bv = arg_to_bv.get(&la);
                    let shorter_bv = arg_to_bv.get(&sa);
                    if let (Some(&lb), Some(&sb)) = (longer_bv, shorter_bv) {
                        if lb != sb {
                            entries.push(tcx.mk_outlives_arg(lb, sb).into());
                        }
                    }
                }
            }
        }
    }

    if entries.is_empty() {
        return None;
    }

    entries.sort_by(|a: &GenericArg<'tcx>, b: &GenericArg<'tcx>| {
        let ao = a.as_outlives().unwrap();
        let bo = b.as_outlives().unwrap();
        (ao.longer(), ao.shorter()).cmp(&(bo.longer(), bo.shorter()))
    });
    entries.dedup();

    let interned = tcx.arena.alloc_from_iter(entries);
    Some(ty::trait_cast::OutlivesClass::from_entries(interned))
}

/// Computes the reflexive-transitive closure of outlives relationships
/// over a `dim`-dimensional index space using Floyd-Warshall.
///
/// `entries` contains `GenericArgKind::Outlives` pairs encoding direct
/// outlives edges. Index `dim - 1` represents `'static` (outlives
/// everything). `usize::MAX` in an `OutlivesArgData` field is remapped
/// to `dim - 1`.
///
/// The resulting `BitMatrix` satisfies: `reach.contains(a, b)` iff
/// lifetime `a` transitively outlives lifetime `b` under the given
/// constraints.
pub(crate) fn outlives_reachability<'tcx>(
    _tcx: TyCtxt<'tcx>,
    (entries, dim): (&'tcx [GenericArg<'tcx>], usize),
) -> BitMatrix<usize, usize> {
    let static_idx = dim - 1;
    let mut reach = BitMatrix::new(dim, dim);

    // Reflexivity.
    for i in 0..dim {
        reach.insert(i, i);
    }
    // 'static outlives everything.
    for j in 0..dim {
        reach.insert(static_idx, j);
    }

    // Direct edges, remapping usize::MAX → static_idx.
    let remap = |idx: usize| if idx == usize::MAX { static_idx } else { idx };
    for entry in entries {
        if let ty::GenericArgKind::Outlives(o) = entry.kind() {
            reach.insert(remap(o.longer()), remap(o.shorter()));
        }
    }

    // Transitive closure (Floyd-Warshall — dim is tiny, typically ≤10).
    for k in 0..dim {
        for i in 0..dim {
            if reach.contains(i, k) {
                reach.union_rows(k, i);
            }
        }
    }

    reach
}

/// Query: is this impl universally admissible — admissible under every
/// outlives class for every dyn binder structure?
///
/// True when the impl satisfies:
/// (a) no concrete lifetimes (e.g. 'static) in the impl's trait ref
/// (b) every trait lifetime position maps to a distinct free impl param
/// (c) no `RegionOutlives` where clauses involving trait lifetime params
/// (d) no trait-position lifetime param also appears in Self
///     (shared params are anchored to the concrete type's erased
///     lifetime, requiring 'static-equivalence)
pub(crate) fn impl_universally_admissible<'tcx>(tcx: TyCtxt<'tcx>, impl_def_id: DefId) -> bool {
    let Some(impl_trait_ref) = tcx.impl_opt_trait_ref(impl_def_id) else {
        return true; // inherent impl => vacuously admissible
    };
    let impl_trait_ref = impl_trait_ref.skip_binder();

    // Walk the impl's trait ref (excluding Self) to check (a)+(b).
    let mut trait_params: UnordSet<EarlyParamRegion> = UnordSet::default();
    let mut ok = true;
    let mut checker =
        UniversalAdmissibilityChecker { seen_params: &mut trait_params, admissible: &mut ok };
    for arg in impl_trait_ref.args.iter().skip(1) {
        let arg: ty::GenericArg<'tcx> = arg;
        arg.visit_with(&mut checker);
        if !*checker.admissible {
            return false;
        }
    }
    drop(checker);

    // Check (d): no trait-position param also appears in Self.
    let mut self_params: UnordSet<EarlyParamRegion> = UnordSet::default();
    let mut collector = SelfRegionCollector { params: &mut self_params };
    impl_trait_ref.self_ty().visit_with(&mut collector);
    drop(collector);
    if self_params.items().any(|param| trait_params.contains(&param)) {
        return false;
    }

    // Check (c): no RegionOutlives where clauses on trait params.
    let predicates = tcx.predicates_of(impl_def_id);
    for (pred, _) in predicates.predicates {
        if let ty::ClauseKind::RegionOutlives(outlives) = pred.kind().skip_binder() {
            let l_is_trait = is_trait_region_param(&trait_params, outlives.0);
            let s_is_trait = is_trait_region_param(&trait_params, outlives.1);
            if l_is_trait || s_is_trait {
                return false;
            }
        }
    }
    true
}

/// Collects all `ReEarlyParam` regions from a type.
struct SelfRegionCollector<'a> {
    params: &'a mut UnordSet<EarlyParamRegion>,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for SelfRegionCollector<'_> {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if let ty::ReEarlyParam(param) = r.kind() {
            self.params.insert(param);
        }
    }
}

/// TypeVisitor that checks conditions (a) and (b).
struct UniversalAdmissibilityChecker<'a> {
    seen_params: &'a mut UnordSet<EarlyParamRegion>,
    admissible: &'a mut bool,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for UniversalAdmissibilityChecker<'_> {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        if !*self.admissible {
            return;
        }
        match r.kind() {
            ty::ReEarlyParam(param) => {
                // (b): each param must appear at most once
                if !self.seen_params.insert(param) {
                    *self.admissible = false;
                }
            }
            ty::ReStatic => *self.admissible = false, // (a)
            ty::ReBound(..) => {}                     // higher-ranked, fine
            _ => *self.admissible = false,            // conservative reject
        }
    }
}

fn is_trait_region_param(
    trait_params: &UnordSet<EarlyParamRegion>,
    region: ty::Region<'_>,
) -> bool {
    match region.kind() {
        ty::ReEarlyParam(param) => trait_params.contains(&param),
        _ => false,
    }
}

// ── TypeVisitor helpers for `impl_admissible_under_class` ─────────────────

/// Extract the dyn type's bound variable index at each trait-lifetime
/// position. Walks the existential binder's trait ref in TypeVisitor
/// DFS order. Returns `Some(bv_index)` for bound regions, `None` for
/// concrete lifetimes (e.g., 'static embedded in a type argument).
pub(crate) fn extract_dyn_bv_positions<'tcx>(
    _tcx: TyCtxt<'tcx>,
    dyn_type: Ty<'tcx>,
) -> Vec<Option<usize>> {
    // dyn_type is `dyn for<'^0, '^1, ...> SubTrait<...>`.
    // Walk the binder's trait ref to find BoundRegion indices.
    let ty::Dynamic(dyn_data, ..) = dyn_type.kind() else {
        bug!("extract_dyn_bv_positions: {dyn_type:?} is not a dyn type");
    };
    let binder = dyn_data.principal().unwrap();
    let mut collector = BoundRegionCollector::new();
    // ExistentialTraitRef.args already excludes Self (erased by
    // `erase_self_ty`), so no skip(1) — iterate all args.
    for arg in binder.skip_binder().args.iter() {
        arg.visit_with(&mut collector);
    }
    collector.into_bv_positions()
}

/// Extract the impl's region at each trait-lifetime position.
/// Walks the impl trait ref's generic args (excluding Self) in the
/// same TypeVisitor DFS order as `extract_dyn_bv_positions`.
///
/// `TraitRef.args[0]` is Self (unlike `ExistentialTraitRef` which
/// already excludes Self), so `.skip(1)` is correct here. Self's
/// regions are handled separately by the Self-anchored-params check
/// in `impl_admissible_under_class`.
pub(crate) fn extract_impl_trait_ref_regions<'tcx>(
    _tcx: TyCtxt<'tcx>,
    trait_ref: ty::TraitRef<'tcx>,
) -> Vec<ty::Region<'tcx>> {
    let mut collector = RegionPositionCollector::new();
    for arg in trait_ref.args.iter().skip(1) {
        arg.visit_with(&mut collector);
    }
    collector.into_regions()
}

/// Collects bound variable indices from a dyn type's existential trait
/// ref regions. Records `Some(bv_index)` for `ReBound` regions and
/// `None` for concrete lifetimes, in TypeVisitor DFS order.
pub(crate) struct BoundRegionCollector {
    positions: Vec<Option<usize>>,
    next_erased_var: usize,
}

impl BoundRegionCollector {
    pub(crate) fn new() -> Self {
        Self { positions: Vec::new(), next_erased_var: 0 }
    }
    pub(crate) fn into_bv_positions(self) -> Vec<Option<usize>> {
        self.positions
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for BoundRegionCollector {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            ty::ReBound(_, br) => self.positions.push(Some(br.var.as_usize())),
            ty::ReErased => {
                let var = self.next_erased_var;
                self.next_erased_var += 1;
                self.positions.push(Some(var));
            }
            _ => self.positions.push(None),
        }
    }
}

/// Collects regions from an impl's trait ref in TypeVisitor DFS order.
pub(crate) struct RegionPositionCollector<'tcx> {
    regions: Vec<ty::Region<'tcx>>,
}

impl<'tcx> RegionPositionCollector<'tcx> {
    pub(crate) fn new() -> Self {
        Self { regions: Vec::new() }
    }
    pub(crate) fn into_regions(self) -> Vec<ty::Region<'tcx>> {
        self.regions
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for RegionPositionCollector<'tcx> {
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        self.regions.push(r);
    }
}

/// Map a region to the set of bv indices it covers, via param_to_bvs.
pub(crate) fn region_to_bvs<'a>(
    param_to_bvs: &'a FxHashMap<RegionVid, DenseBitSet<usize>>,
    region: ty::Region<'_>,
) -> Option<&'a DenseBitSet<usize>> {
    match region.kind() {
        ty::ReEarlyParam(param) => param_to_bvs.get(&RegionVid::from_u32(param.index)),
        _ => None,
    }
}

/// Resolve whether a concrete type fully satisfies a dyn type —
/// both its principal trait and all auto trait bounds (e.g., `Send`,
/// `Sync`). Returns the principal impl's `DefId` if all obligations
/// are met, `None` otherwise.
///
/// Uses `codegen_select_candidate` — the same path used by vtable
/// computation in unsizing coercions.
///
/// The dyn type's binder may contain `ReBound` regions (e.g.,
/// `dyn for<'a, 'b> Sub<'a, 'b>`). These appear in the constructed
/// `TraitRef` but are harmless: `codegen_select_candidate` builds its
/// `InferCtxt` with `.ignoring_regions()`, so they unify freely with
/// impl params. We only need the `DefId`; lifetime admissibility is
/// checked separately by `impl_admissible_under_class`.
///
/// **Auto traits.** The dyn type may carry non-principal auto trait
/// bounds (e.g., `dyn Sub + Send`). These are checked separately from
/// the principal: for each auto trait `DefId` in the dyn type, the
/// function verifies that `concrete_ty: AutoTrait` holds. If any auto
/// trait is unsatisfied, the concrete type cannot soundly inhabit a
/// trait object of this dyn type, and `None` is returned — even if
/// the principal trait is implemented.
pub(crate) fn resolve_dyn_satisfaction<'tcx>(
    tcx: TyCtxt<'tcx>,
    concrete_ty: Ty<'tcx>,
    sub_trait_dyn: Ty<'tcx>,
) -> Option<DefId> {
    let ty::Dynamic(dyn_data, ..) = sub_trait_dyn.kind() else {
        bug!("resolve_dyn_satisfaction: {sub_trait_dyn:?} is not a dyn type");
    };
    let principal = dyn_data.principal()?;
    let existential_ref = principal.skip_binder();

    let typing_env = ty::TypingEnv::fully_monomorphized();

    // --- Check auto trait bounds ---
    //
    // Auto traits (Send, Sync, Unpin, etc.) have no methods and no
    // vtable entries, but they are semantic obligations on the dyn
    // type. A table entry must be `None` if the concrete type fails
    // any auto trait bound, regardless of principal satisfaction.
    for auto_trait_def_id in dyn_data.auto_traits() {
        let auto_ref = ty::TraitRef::new(tcx, auto_trait_def_id, [concrete_ty]);
        let input = typing_env.as_query_input(auto_ref);
        match tcx.codegen_select_candidate(input) {
            Ok(_) => {}            // Satisfied.
            Err(_) => return None, // Auto trait not implemented.
        }
    }

    // --- Check principal trait ---
    //
    // ExistentialTraitRef.args already excludes Self, so prepend
    // concrete_ty as Self — equivalent to
    // `existential_ref.with_self_ty(tcx, concrete_ty)`.
    let concrete_trait_ref = ty::TraitRef::new(
        tcx,
        existential_ref.def_id,
        std::iter::once(concrete_ty.into()).chain(existential_ref.args.iter()),
    );

    debug_assert_eq!(
        concrete_trait_ref,
        tcx.normalize_erasing_regions(typing_env, ty::Unnormalized::new_wip(concrete_trait_ref)),
        "resolve_dyn_satisfaction: trait ref must be normalized post-mono",
    );
    let input = typing_env.as_query_input(concrete_trait_ref);

    match tcx.codegen_select_candidate(input) {
        Ok(ImplSource::UserDefined(ImplSourceUserDefinedData { impl_def_id, .. })) => {
            Some(*impl_def_id)
        }
        Ok(_) => None, // Builtin or Param — no user impl.
        Err(CodegenObligationError::Ambiguity | CodegenObligationError::Unimplemented) => None,
        Err(CodegenObligationError::UnconstrainedParam(_)) => {
            bug!(
                "resolve_dyn_satisfaction: unconstrained param in impl \
                 for `{concrete_ty}: {:?}`",
                existential_ref.def_id,
            );
        }
    }
}
