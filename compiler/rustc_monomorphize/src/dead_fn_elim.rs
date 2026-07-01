//! Dead function elimination via BFS reachability — `-Z dead-fn-elimination`.
//!
//! Identifies functions unreachable from the binary entry point via a breadth-first
//! search over the call graph, then marks them so they can be removed from the set of
//! monomorphized items before codegen.
//!
//! This pass is scoped to binary crates only (executables and `bin` test harnesses).
//! Library crate types early-return at the top of [`run_analysis`] because their public
//! items are reachable by definition (downstream crates may call any `pub fn`) and there
//! is no entry point to seed the search from.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::panic;

use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::def::DefKind;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::{Body, TerminatorKind};
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::DefId;

// `thread_local!` keeps the footprint of this experimental flag minimal. If the flag is
// stabilized, this state should move to a `GlobalCtxt` field or a dedicated query so the
// reachable set is computed and cached through the normal query machinery.
thread_local! {
    /// Trait `DefId` indices seen in vtable constructions (`dyn Trait` casts).
    static VTABLE_TRAITS: RefCell<FxHashSet<u64>> = RefCell::new(FxHashSet::default());

    /// Function `DefId`s whose address has been taken. Populated during MIR traversal by
    /// [`scan_for_address_taken`]. These are unioned into the BFS seeds so they survive the
    /// call graph: an indirect call site is not visible to direct-call BFS.
    static ADDRESS_TAKEN: RefCell<FxHashSet<u64>> = RefCell::new(FxHashSet::default());

    /// Functions identified as unreachable and safe to eliminate. Populated by
    /// [`run_analysis`]; read via [`is_eliminable`].
    static ELIMINABLE_DEF_IDS: RefCell<FxHashSet<u64>> = RefCell::new(FxHashSet::default());
}

/// Encode a `DefId` as a `u64` key: high 32 bits = crate index, low 32 bits = item index.
#[inline]
fn def_id_key(def_id: DefId) -> u64 {
    ((def_id.krate.as_u32() as u64) << 32) | (def_id.index.as_u32() as u64)
}

/// Run BFS reachability analysis and populate `ELIMINABLE_DEF_IDS`.
///
/// `mono_items` is the set of monomorphized items already collected by the caller; it is
/// used as the ground truth for [`is_safe_to_eliminate`], so this function never re-enters
/// the monomorphization collection query.
pub(crate) fn run_analysis(tcx: TyCtxt<'_>, mono_items: &[MonoItem<'_>]) {
    // Skip analysis entirely for library crates: we never eliminate public items, so the
    // only candidates would be private functions — but without a binary entry point there
    // are no seeds, meaning every private function would be eliminated incorrectly.
    // Returning early is safe and avoids MIR traversal overhead.
    if tcx.entry_fn(()).is_none() {
        return;
    }

    // Build the set of `DefId`s that the collector kept. Dropping any of these would
    // dangle a symbol, so they are never eligible for elimination.
    let mono_def_ids: FxHashSet<DefId> = mono_items
        .iter()
        .filter_map(|item| match item {
            MonoItem::Fn(instance) => Some(instance.def_id()),
            MonoItem::Static(def_id) => Some(*def_id),
            _ => None,
        })
        .collect();

    // Build the call graph (vtable constructions are scanned inline by `add_mir_edges`,
    // which also records address-taken functions).
    let call_graph = build_call_graph(tcx);

    // BFS from entry seeds. Vtable traits and address-taken functions were populated by
    // `build_call_graph`; the latter are unioned into the seed set below so that functions
    // reached only through indirect calls survive.
    let mut seeds = collect_seeds(tcx);
    // Iteration order over the `FxHashSet` does not affect the resulting seed set.
    #[allow(rustc::potential_query_instability)]
    ADDRESS_TAKEN.with(|a| {
        for &key in a.borrow().iter() {
            seeds.insert(key);
        }
    });
    let reachable = run_bfs(seeds, &call_graph);

    // Every item in `reachable_set` was unioned into the seeds (via `collect_seeds`), so
    // BFS cannot drop it. This invariant guards against future regressions.
    #[cfg(debug_assertions)]
    {
        let reachable_set = tcx.reachable_set(());
        for &local_def_id in tcx.mir_keys(()) {
            if reachable_set.contains(&local_def_id) {
                let key = def_id_key(local_def_id.to_def_id());
                debug_assert!(
                    reachable.contains(&key),
                    "invariant violated: reachable_set item {key:?} missing from post-BFS set"
                );
            }
        }
    }

    // Mark unreachable and safe functions as eliminable.
    let mut count = 0usize;
    ELIMINABLE_DEF_IDS.with(|e| {
        let mut eliminable = e.borrow_mut();
        for &local_def_id in tcx.mir_keys(()) {
            let def_id = local_def_id.to_def_id();
            if !tcx.def_kind(def_id).is_fn_like() {
                continue;
            }
            let key = def_id_key(def_id);
            if !reachable.contains(&key) && is_safe_to_eliminate(tcx, def_id, &mono_def_ids) {
                eliminable.insert(def_id.index.as_u32() as u64);
                count += 1;
            }
        }
    });

    if count > 0 {
        tcx.sess.dcx().note(format!(
            "{count} unreachable functions excluded from codegen by -Z dead-fn-elimination"
        ));
    }
}

/// Returns `true` if the given local `DefId` index was identified as eliminable. Callers
/// use this for per-item O(1) lookups, so it avoids cloning the set.
pub(crate) fn is_eliminable(idx: u64) -> bool {
    ELIMINABLE_DEF_IDS.with(|e| e.borrow().contains(&idx))
}

/// Collect the BFS seed set: everything that must be kept regardless of the call graph.
fn collect_seeds(tcx: TyCtxt<'_>) -> FxHashSet<u64> {
    let mut seeds = FxHashSet::default();

    // `reachable_set` is the non-eliminable lower bound. It already covers public items by
    // effective visibility, lang items, trait impl items, custom-linkage symbols
    // (`#[no_mangle]`/`#[used]`/`#[export_name]`), foreign-item-symbol aliases, and
    // (transitively) const-initializer references — including the test-harness `TESTS`
    // slice. See `compiler/rustc_passes/src/reachable.rs`.
    let reachable_set = tcx.reachable_set(());
    for &local_def_id in tcx.mir_keys(()) {
        if reachable_set.contains(&local_def_id) {
            seeds.insert(def_id_key(local_def_id.to_def_id()));
        }
    }

    // The entry function is a binary-specific seed that `reachable_set` does not provide.
    if let Some((entry_def_id, _)) = tcx.entry_fn(()) {
        seeds.insert(def_id_key(entry_def_id));
    }

    // Vtable-constructed trait impl methods (local). Dynamic dispatch is invisible to the
    // call-graph BFS, so methods of any trait used as `dyn Trait` must be seeded.
    for &local_def_id in tcx.mir_keys(()) {
        let def_id = local_def_id.to_def_id();
        if tcx.def_kind(def_id) != DefKind::AssocFn {
            continue;
        }
        let assoc_item = tcx.associated_item(def_id);
        if let Some(trait_method_def_id) = assoc_item.trait_item_def_id() {
            let trait_def_id = tcx.parent(trait_method_def_id);
            if tcx.is_dyn_compatible(trait_def_id) {
                let trait_idx = trait_def_id.index.as_u32() as u64;
                if VTABLE_TRAITS.with(|v| v.borrow().contains(&trait_idx)) {
                    seeds.insert(def_id_key(def_id));
                }
            }
        }
    }
    seeds
}

/// Build call graph edges for all local (same-crate) fn-like items.
///
/// Cross-crate edges are intentionally not added: `is_safe_to_eliminate` rejects any
/// non-local `DefId`, so extern-crate edges could never demote an item from the eliminable
/// set. Cross-crate effects are captured instead by `reachable_set` and address-taken
/// seeding.
fn build_call_graph(tcx: TyCtxt<'_>) -> FxIndexMap<u64, FxIndexSet<u64>> {
    let mut graph: FxIndexMap<u64, FxIndexSet<u64>> = FxIndexMap::default();
    for &local_def_id in tcx.mir_keys(()) {
        let def_id = local_def_id.to_def_id();
        let def_kind = tcx.def_kind(def_id);
        if def_kind.is_fn_like() {
            add_mir_edges(tcx, def_id, &mut graph);
        } else if matches!(
            def_kind,
            DefKind::Const { .. } | DefKind::Static { .. } | DefKind::AssocConst { .. }
        ) {
            // A `const FLAGS: &[&dyn Flag] = &[..]` performs its `Unsize` coercions inside
            // a const/static initializer rather than in function-body MIR. By the time we
            // read it the coercions are const-evaluated into an allocation whose provenance
            // points at `GlobalAlloc::VTable`/`GlobalAlloc::Function`, which
            // `scan_for_address_taken` (it only walks fn MIR) misses. We must walk the
            // evaluated allocation graph to seed `VTABLE_TRAITS` and `ADDRESS_TAKEN`,
            // otherwise impl methods reachable only through such const vtables would be
            // wrongly eliminated.
            scan_const_static_allocs(tcx, def_id);
        }
    }
    graph
}

/// Read MIR for `def_id` and insert call edges into `graph`. Silently skips if MIR is
/// unavailable or if `optimized_mir` panics.
fn add_mir_edges(tcx: TyCtxt<'_>, def_id: DefId, graph: &mut FxIndexMap<u64, FxIndexSet<u64>>) {
    if !tcx.is_mir_available(def_id) {
        return;
    }
    let Ok(body) = panic::catch_unwind(panic::AssertUnwindSafe(|| tcx.optimized_mir(def_id)))
    else {
        return;
    };
    scan_for_address_taken(body);
    let caller_key = def_id_key(def_id);
    let edges = graph.entry(caller_key).or_default();
    for bb in body.basic_blocks.iter() {
        // Direct call edges.
        if let TerminatorKind::Call { func, .. } = &bb.terminator().kind {
            if let rustc_middle::mir::Operand::Constant(c) = func {
                if let ty::FnDef(callee_def_id, _) = c.const_.ty().kind() {
                    edges.insert(def_id_key(*callee_def_id));
                }
            }
        }
        // A closure/coroutine is created via an `Aggregate` rvalue and invoked later
        // through `Fn*`/`poll`, so there is no direct `Call` to it from its creator;
        // without this edge the closure body (and everything it calls) would be wrongly
        // unreachable. Likewise any `FnDef` mentioned as a value (passed to an adaptor such
        // as `iter.map(f)`) must become an edge, not only direct call targets.
        for stmt in &bb.statements {
            use rustc_middle::mir::{AggregateKind, Rvalue, StatementKind};
            if let StatementKind::Assign(box (_, rvalue)) = &stmt.kind {
                // Closure / coroutine / coroutine-closure construction.
                if let Rvalue::Aggregate(box kind, _) = rvalue {
                    match kind {
                        AggregateKind::Closure(cdef, _)
                        | AggregateKind::Coroutine(cdef, _)
                        | AggregateKind::CoroutineClosure(cdef, _) => {
                            edges.insert(def_id_key(*cdef));
                        }
                        _ => {}
                    }
                }
                // Any operand that names a function by value (a fn item passed around).
                for op in rvalue_operands(rvalue) {
                    if let rustc_middle::mir::Operand::Constant(c) = op {
                        if let ty::FnDef(fdef, _) = c.const_.ty().kind() {
                            edges.insert(def_id_key(*fdef));
                        }
                    }
                }
            }
        }
    }
}

/// Best-effort iterator over the operands an `Rvalue` reads, so we can spot `FnDef` values
/// used as arguments to adaptors (`iter.map(some_fn)`), not only direct call targets.
fn rvalue_operands<'a, 'tcx>(
    rvalue: &'a rustc_middle::mir::Rvalue<'tcx>,
) -> Vec<&'a rustc_middle::mir::Operand<'tcx>> {
    use rustc_middle::mir::Rvalue;
    let mut out = Vec::new();
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::Repeat(op, _)
        | Rvalue::Cast(_, op, _)
        | Rvalue::UnaryOp(_, op) => out.push(op),
        Rvalue::BinaryOp(_, box (a, b)) => {
            out.push(a);
            out.push(b);
        }
        Rvalue::Aggregate(_, ops) => out.extend(ops.iter()),
        _ => {}
    }
    out
}

/// Walk a const/static's evaluated allocation graph, recording any vtable traits
/// (`GlobalAlloc::VTable`) and address-taken functions (`GlobalAlloc::Function`) reachable
/// through its provenance. Silently skips on any evaluation failure.
fn scan_const_static_allocs(tcx: TyCtxt<'_>, def_id: DefId) {
    use rustc_middle::mir::interpret::GlobalAlloc;

    let mut roots: Vec<rustc_middle::mir::interpret::AllocId> = Vec::new();

    // Statics and consts use different queries: `const_eval_poly` asserts it is not called
    // on a static (statics are places, not values).
    if tcx.is_static(def_id) {
        if let Ok(Ok(alloc)) =
            panic::catch_unwind(panic::AssertUnwindSafe(|| tcx.eval_static_initializer(def_id)))
        {
            collect_alloc_ids(alloc, &mut roots);
        }
    } else if matches!(tcx.def_kind(def_id), DefKind::Const { .. } | DefKind::AssocConst { .. }) {
        // Only non-generic consts are poly-evaluable; generic ones panic the query.
        if !tcx.generics_of(def_id).requires_monomorphization(tcx) {
            if let Ok(Ok(val)) =
                panic::catch_unwind(panic::AssertUnwindSafe(|| tcx.const_eval_poly(def_id)))
            {
                push_const_value_allocs(val, &mut roots);
            }
        }
    }

    // BFS over the allocation graph following provenance pointers.
    let mut seen: FxHashSet<rustc_middle::mir::interpret::AllocId> = FxHashSet::default();
    while let Some(id) = roots.pop() {
        if !seen.insert(id) {
            continue;
        }
        match tcx.try_get_global_alloc(id) {
            Some(GlobalAlloc::VTable(_ty, predicates)) => {
                if let Some(principal) = predicates.principal() {
                    let trait_def_id = principal.def_id();
                    VTABLE_TRAITS.with(|v| {
                        v.borrow_mut().insert(trait_def_id.index.as_u32() as u64);
                    });
                }
            }
            Some(GlobalAlloc::Function { instance }) => {
                ADDRESS_TAKEN.with(|a| {
                    a.borrow_mut().insert(def_id_key(instance.def_id()));
                });
            }
            Some(GlobalAlloc::Memory(alloc)) => {
                collect_alloc_ids(alloc, &mut roots);
            }
            // A reference to another static — its own scan covers it.
            _ => {}
        }
    }
}

/// Push every `AllocId` referenced by `alloc`'s provenance into `out`.
fn collect_alloc_ids(
    alloc: rustc_middle::mir::interpret::ConstAllocation<'_>,
    out: &mut Vec<rustc_middle::mir::interpret::AllocId>,
) {
    for prov in alloc.inner().provenance().provenances() {
        out.push(prov.alloc_id());
    }
}

/// Extract any `AllocId`s carried by a `ConstValue` (`Indirect`/`Slice` carry one directly;
/// a thin pointer is `Scalar(Scalar::Ptr)`).
fn push_const_value_allocs(
    val: rustc_middle::mir::ConstValue,
    out: &mut Vec<rustc_middle::mir::interpret::AllocId>,
) {
    use rustc_middle::mir::ConstValue;
    use rustc_middle::mir::interpret::Scalar;
    match val {
        ConstValue::Indirect { alloc_id, .. } | ConstValue::Slice { alloc_id, .. } => {
            out.push(alloc_id);
        }
        ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
            out.push(ptr.provenance.alloc_id());
        }
        _ => {}
    }
}

/// BFS over the call graph from the seed set, returning the set of reachable keys.
fn run_bfs(seeds: FxHashSet<u64>, graph: &FxIndexMap<u64, FxIndexSet<u64>>) -> FxHashSet<u64> {
    let mut marked: FxHashSet<u64> = FxHashSet::default();
    let mut queue = VecDeque::new();
    // Seeds are a membership-only set; sort them for deterministic traversal order.
    #[allow(rustc::potential_query_instability)]
    let mut sorted_seeds: Vec<u64> = seeds.into_iter().collect();
    sorted_seeds.sort_unstable();
    for seed in sorted_seeds {
        if marked.insert(seed) {
            queue.push_back(seed);
        }
    }
    while let Some(current) = queue.pop_front() {
        if let Some(callees) = graph.get(&current) {
            // `FxIndexSet` preserves insertion order; iterate directly.
            for &callee in callees {
                if marked.insert(callee) {
                    queue.push_back(callee);
                }
            }
        }
    }
    marked
}

/// Safety check combining the cheap local floor with the `mono_items` ground truth: if the
/// monomorphization collector kept this item, dropping it would dangle a symbol.
fn is_safe_to_eliminate(tcx: TyCtxt<'_>, def_id: DefId, mono_items: &FxHashSet<DefId>) -> bool {
    if mono_items.contains(&def_id) {
        return false;
    }
    is_locally_safe_to_eliminate(tcx, def_id)
}

/// Cheap, local-only safety floor (no monomorphization collection). Rejects items that are
/// unsound to drop regardless of reachability: non-local, non-fn-like, vtable-trait impl
/// methods, linker-visible symbols, `Drop` glue, the entry fn, generics, and async.
fn is_locally_safe_to_eliminate(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if !def_id.is_local() {
        return false;
    }

    let def_kind = tcx.def_kind(def_id);
    match def_kind {
        DefKind::Fn => {}
        DefKind::AssocFn => {
            // Don't eliminate trait impl methods for vtable-constructed traits.
            let assoc_item = tcx.associated_item(def_id);
            if let Some(trait_method_def_id) = assoc_item.trait_item_def_id() {
                let trait_def_id = tcx.parent(trait_method_def_id);
                if tcx.is_dyn_compatible(trait_def_id) {
                    let trait_idx = trait_def_id.index.as_u32() as u64;
                    if VTABLE_TRAITS.with(|v| v.borrow().contains(&trait_idx)) {
                        return false;
                    }
                }
            }
        }
        _ => return false,
    }

    // Don't eliminate linker-visible symbols.
    let attrs = tcx.codegen_fn_attrs(def_id);
    if attrs.flags.intersects(CodegenFnAttrFlags::NO_MANGLE | CodegenFnAttrFlags::USED_LINKER)
        || attrs.symbol_name.is_some()
        || attrs.linkage.is_some()
    {
        return false;
    }

    // Don't eliminate `Drop::drop` implementations.
    if def_kind == DefKind::AssocFn {
        if let Some(drop_trait) = tcx.lang_items().drop_trait() {
            let assoc_item = tcx.associated_item(def_id);
            if let Some(trait_item_id) = assoc_item.trait_item_def_id() {
                if tcx.parent(trait_item_id) == drop_trait {
                    return false;
                }
            }
        }
    }

    // Don't eliminate the entry function.
    if tcx.entry_fn(()).map(|(id, _)| id) == Some(def_id) {
        return false;
    }

    // Don't eliminate generic functions: they are not codegened until monomorphized, and
    // `requires_monomorphization` correctly singles out type- and const-generic parameters
    // (lifetime parameters do not require monomorphization).
    if tcx.generics_of(def_id).requires_monomorphization(tcx) {
        return false;
    }

    // Don't eliminate async functions (the state-machine transform runs after MIR).
    if tcx.asyncness(def_id).is_async() {
        return false;
    }

    true
}

/// Scan a MIR body for address-taking operations. Records vtable constructions
/// (`PointerCoercion::Unsize` to `dyn Trait`) into `VTABLE_TRAITS`, and function-pointer
/// reifications / closure-to-fn-pointer coercions into `ADDRESS_TAKEN`. Inline-asm `sym fn`
/// operands are handled by [`scan_inline_asm`].
fn scan_for_address_taken(body: &Body<'_>) {
    use rustc_middle::mir::{CastKind, Operand, Rvalue, StatementKind};
    use rustc_middle::ty::adjustment::PointerCoercion;
    for bb in body.basic_blocks.iter() {
        for stmt in &bb.statements {
            if let StatementKind::Assign(box (_, Rvalue::Cast(kind, op, target_ty))) = &stmt.kind {
                match kind {
                    CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                        record_dyn_traits(*target_ty);
                    }
                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _)
                    | CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _) => {
                        if let Operand::Constant(c) = op {
                            if let ty::FnDef(def_id, _) = c.const_.ty().kind() {
                                ADDRESS_TAKEN.with(|a| {
                                    a.borrow_mut().insert(def_id_key(*def_id));
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        scan_inline_asm(&bb.terminator().kind);
    }
}

/// Inline-asm `sym fn` operands reference functions outside the call graph.
fn scan_inline_asm(kind: &TerminatorKind<'_>) {
    if let TerminatorKind::InlineAsm { operands, .. } = kind {
        for op in operands.iter() {
            if let rustc_middle::mir::InlineAsmOperand::SymFn { value } = op {
                if let ty::FnDef(def_id, _) = value.const_.ty().kind() {
                    ADDRESS_TAKEN.with(|a| {
                        a.borrow_mut().insert(def_id_key(*def_id));
                    });
                }
            }
        }
    }
}

/// Record the principal trait of any `dyn Trait` type into `VTABLE_TRAITS`.
fn record_dyn_traits(ty: Ty<'_>) {
    match ty.kind() {
        ty::Dynamic(predicates, ..) => {
            if let Some(def_id) = predicates.principal_def_id() {
                VTABLE_TRAITS.with(|v| {
                    v.borrow_mut().insert(def_id.index.as_u32() as u64);
                });
            }
        }
        ty::Ref(_, inner, _) | ty::RawPtr(inner, _) => record_dyn_traits(*inner),
        _ => {}
    }
}
