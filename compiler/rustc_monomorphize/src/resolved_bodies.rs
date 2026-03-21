//! Synthetic MIR body construction for resolved trait-cast intrinsics.

use rustc_middle::mir::interpret::{AllocId, Pointer, Scalar};
use rustc_middle::mir::{
    BasicBlock, Body, Const, ConstOperand, ConstValue, Operand, Place, Rvalue, Statement,
    StatementKind, TerminatorKind,
};
use rustc_middle::ty::trait_cast::{IntrinsicResolutions, OutlivesClass};
use rustc_middle::ty::{self, GenericArgsRef, Instance, Ty, TyCtxt, TypeFoldable};
use rustc_span::{DUMMY_SP, sym};

use crate::cast_sensitivity::{CallerOutlivesEnv, compose_all_through_chain};
use crate::erasure_safe::{
    region_slots_of_ty, remap_trait_metadata_outlives_entries_from_origin_positions,
};
use crate::trait_graph::derive_where_clause_outlives_class;

// ── Rvalue construction helpers ─────────────────────────────────────────────

/// Build a constant `Rvalue` for a resolved `trait_metadata_index` intrinsic.
///
/// Returns `Rvalue::Aggregate(Tuple, [&'static u8, usize])` where the first
/// field is a reference to the global crate ID allocation and the second is
/// the resolved table index.
fn build_index_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    global_crate_id: AllocId,
    index: usize,
) -> Rvalue<'tcx> {
    let ref_operand = make_static_ref_operand(tcx, global_crate_id);
    let index_operand = make_usize_operand(tcx, index);

    Rvalue::Aggregate(
        Box::new(rustc_middle::mir::AggregateKind::Tuple),
        [ref_operand, index_operand].into_iter().collect(),
    )
}

/// Build a constant `Rvalue` for a resolved `trait_metadata_table` intrinsic.
///
/// Returns `Rvalue::Aggregate(Tuple, [&'static u8, NonNull<...>])` where the
/// first field is a reference to the global crate ID allocation and the second
/// is a `NonNull` pointer to the metadata table static.
///
/// `NonNull<T>` is a `#[repr(transparent)]` wrapper around `*const T`, so at
/// the MIR level it is represented as a raw pointer scalar.
fn build_table_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    global_crate_id: AllocId,
    table_alloc: AllocId,
    return_ty: Ty<'tcx>,
) -> Rvalue<'tcx> {
    let ref_operand = make_static_ref_operand(tcx, global_crate_id);

    // The NonNull field type is the second field of the return tuple.
    let nonnull_ty = extract_tuple_field_ty(return_ty, 1);
    let ptr = Pointer::from(table_alloc);
    let scalar = Scalar::from_pointer(ptr, &tcx);
    let nonnull_operand = Operand::Constant(Box::new(ConstOperand {
        span: DUMMY_SP,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(scalar), nonnull_ty),
    }));

    Rvalue::Aggregate(
        Box::new(rustc_middle::mir::AggregateKind::Tuple),
        [ref_operand, nonnull_operand].into_iter().collect(),
    )
}

/// Build a constant `Rvalue` for a resolved `trait_metadata_table_len` intrinsic.
///
/// Returns `Rvalue::Use(Operand::Constant(usize))`.
fn build_table_len_rvalue<'tcx>(tcx: TyCtxt<'tcx>, len: usize) -> Rvalue<'tcx> {
    Rvalue::Use(make_usize_operand(tcx, len))
}

/// Build a constant `Rvalue` for a resolved `trait_cast_is_lifetime_erasure_safe`
/// intrinsic.
///
/// Returns `Rvalue::Use(Operand::Constant(bool))`.
fn build_erasure_safe_rvalue<'tcx>(tcx: TyCtxt<'tcx>, safe: bool) -> Rvalue<'tcx> {
    Rvalue::Use(Operand::Constant(Box::new(ConstOperand {
        span: DUMMY_SP,
        user_ty: None,
        const_: Const::from_bool(tcx, safe),
    })))
}

// ── Operand construction primitives ─────────────────────────────────────────

/// Create a constant operand holding a `&'static u8` reference to the given
/// allocation.
fn make_static_ref_operand<'tcx>(tcx: TyCtxt<'tcx>, alloc_id: AllocId) -> Operand<'tcx> {
    let ref_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_static, tcx.types.u8);
    let ptr = Pointer::from(alloc_id);
    let scalar = Scalar::from_pointer(ptr, &tcx);
    Operand::Constant(Box::new(ConstOperand {
        span: DUMMY_SP,
        user_ty: None,
        const_: Const::Val(ConstValue::Scalar(scalar), ref_ty),
    }))
}

/// Create a constant operand holding a `usize` value.
fn make_usize_operand<'tcx>(tcx: TyCtxt<'tcx>, value: usize) -> Operand<'tcx> {
    Operand::Constant(Box::new(ConstOperand {
        span: DUMMY_SP,
        user_ty: None,
        const_: Const::Val(ConstValue::from_target_usize(value as u64, &tcx), tcx.types.usize),
    }))
}

/// Extract the type of the `idx`-th field from a tuple type.
fn extract_tuple_field_ty<'tcx>(tuple_ty: Ty<'tcx>, idx: usize) -> Ty<'tcx> {
    match tuple_ty.kind() {
        ty::Tuple(fields) => fields[idx],
        _ => {
            rustc_middle::bug!("extract_tuple_field_ty: expected tuple type, got {tuple_ty:?}");
        }
    }
}

// ── Monomorphization ────────────────────────────────────────────────────────

/// Monomorphize a value through the delayed instance's substitution.
///
/// The MIR body obtained from `instance_mir` is generic — types in
/// func operands are expressed in terms of the defining function's
/// generic parameters. This substitutes the delayed instance's args
/// to produce fully concrete types for `IntrinsicResolutions` lookups.
fn monomorphize<'tcx, T>(tcx: TyCtxt<'tcx>, delayed_instance: Instance<'tcx>, value: T) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    delayed_instance.instantiate_mir_and_normalize_erasing_regions(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        ty::EarlyBinder::bind(value),
    )
}

fn identity_outlives_for_origin_positions<'tcx>(
    tcx: TyCtxt<'tcx>,
    summary: &'tcx rustc_middle::mir::BorrowckRegionSummary,
    mapping: &rustc_middle::mir::CallSiteRegionMapping,
    origin_positions: &[Option<usize>],
) -> &'tcx [ty::GenericArg<'tcx>] {
    let caller_env = CallerOutlivesEnv::from_region_summary_walk_pos(tcx, summary, mapping);
    let mut positions: Vec<usize> = origin_positions.iter().flatten().copied().collect();
    positions.sort_unstable();
    positions.dedup();

    let mut entries = Vec::new();
    for &longer in &positions {
        for &shorter in &positions {
            if longer != shorter && caller_env.outlives(longer, shorter) {
                entries.push(tcx.mk_outlives_arg(longer, shorter).into());
            }
        }
    }
    tcx.arena.alloc_from_iter(entries)
}

// ── Table-dependent intrinsic resolution ────────────────────────────────────

/// Given a Call terminator's `func` operand, check whether it is a
/// table-dependent trait-cast intrinsic (`trait_metadata_index`,
/// `trait_metadata_table`, `trait_metadata_table_len`) and, if so,
/// return the resolved constant as an `Rvalue`.
///
/// For `trait_metadata_index`, the `OutlivesClass` is computed via
/// `augmented_outlives_for_call` — the func operand's generic args do
/// NOT carry the caller's outlives environment; that lives on the
/// `delayed_instance` and must be projected through the `call_id` chain.
///
/// `trait_cast_is_lifetime_erasure_safe` is **not** handled here — it
/// is resolved per call site via [`resolve_erasure_safe_callee`].
fn resolve_table_callee<'tcx>(
    tcx: TyCtxt<'tcx>,
    resolutions: &IntrinsicResolutions<'tcx>,
    delayed_instance: Instance<'tcx>,
    call_id: &'tcx ty::List<(rustc_hir::def_id::DefId, u32, GenericArgsRef<'tcx>)>,
    func: &Operand<'tcx>,
) -> Option<Rvalue<'tcx>> {
    let (def_id, generic_args) = func.const_fn_def()?;
    let intrinsic = tcx.intrinsic(def_id)?;

    // The MIR body is generic — generic_args are in the defining
    // function's parameter space. Monomorphize to get concrete types
    // that match the IntrinsicResolutions keys.
    let mono_args: GenericArgsRef<'tcx> = monomorphize(tcx, delayed_instance, generic_args);

    match intrinsic.name {
        s if s == sym::trait_metadata_index => {
            let super_trait = mono_args[0].expect_ty();
            let sub_trait = mono_args[1].expect_ty();
            let callee_instance = Instance::expect_resolve(
                tcx,
                ty::TypingEnv::fully_monomorphized(),
                def_id,
                mono_args,
                DUMMY_SP,
            );
            let mut call_site_outlives =
                tcx.augmented_outlives_for_call((delayed_instance, call_id, callee_instance));
            let total_transport_slots =
                region_slots_of_ty(super_trait) + region_slots_of_ty(sub_trait);
            let origin_positions =
                compose_all_through_chain(tcx, delayed_instance, call_id, total_transport_slots);
            // (Fallback handled below via where-clause class.)
            if !call_site_outlives.is_empty() {
                call_site_outlives = remap_trait_metadata_outlives_entries_from_origin_positions(
                    tcx,
                    super_trait,
                    sub_trait,
                    &origin_positions,
                    call_site_outlives,
                );
            }
            let outlives_class = OutlivesClass::from_entries(call_site_outlives);
            let empty_class = OutlivesClass::from_entries(&[]);
            // Prefer the where-clause-derived class over the empty
            // class. The empty class slot may have a null table entry
            // for traits with where-clause constraints (the empty
            // class can't prove the constraint). The where-clause
            // class represents the minimum evidence the trait requires;
            // the erasure safety check verifies the caller satisfies it.
            let wc_class = derive_where_clause_outlives_class(tcx, sub_trait);
            let index = resolutions
                .indices
                .get(&(sub_trait, outlives_class))
                .and_then(|idx| {
                    // If the CRL class matched directly, use it.
                    if !outlives_class.entries.is_empty() {
                        return Some(idx);
                    }
                    // Empty CRL class — prefer the where-clause class
                    // if available (its slot may be populated while the
                    // empty class slot is null).
                    wc_class
                        .as_ref()
                        .and_then(|wc| resolutions.indices.get(&(sub_trait, *wc)))
                        .or(Some(idx))
                })
                .or_else(|| resolutions.indices.get(&(sub_trait, empty_class)));
            let &index = index?;
            Some(build_index_rvalue(tcx, resolutions.global_crate_id, index))
        }
        s if s == sym::trait_metadata_table => {
            let super_trait = mono_args[0].expect_ty();
            let concrete_type = mono_args[1].expect_ty();
            let &table_alloc = resolutions.tables.get(&(super_trait, concrete_type))?;
            let return_ty = monomorphize(tcx, delayed_instance, func.constant()?.const_.ty());
            let output_ty = return_ty.fn_sig(tcx).output().skip_binder();
            Some(build_table_rvalue(tcx, resolutions.global_crate_id, table_alloc, output_ty))
        }
        s if s == sym::trait_metadata_table_len => {
            let super_trait = mono_args[0].expect_ty();
            let &len = resolutions.table_lens.get(&super_trait)?;
            Some(build_table_len_rvalue(tcx, len))
        }
        _ => None,
    }
}

// ── Erasure-safe resolution ─────────────────────────────────────────────────

/// Check if `func` is a `trait_cast_is_lifetime_erasure_safe` call; if so,
/// compute per-call-site outlives entries via the `augmented_outlives_for_call`
/// query and resolve via the `is_lifetime_erasure_safe` query.
///
/// The `call_id` chain on the Call terminator identifies the inlining path,
/// which determines how the delayed Instance's outlives environment maps into
/// the intrinsic's walk-position space.
fn resolve_erasure_safe_callee<'tcx>(
    tcx: TyCtxt<'tcx>,
    delayed_instance: Instance<'tcx>,
    call_id: &'tcx ty::List<(
        rustc_hir::def_id::DefId,
        u32,
        rustc_middle::ty::GenericArgsRef<'tcx>,
    )>,
    func: &Operand<'tcx>,
) -> Option<Rvalue<'tcx>> {
    let (def_id, generic_args) = func.const_fn_def()?;
    let intrinsic = tcx.intrinsic(def_id)?;
    if intrinsic.name != sym::trait_cast_is_lifetime_erasure_safe {
        return None;
    }

    // Monomorphize generic args to get concrete dyn types.
    let mono_args: GenericArgsRef<'tcx> = monomorphize(tcx, delayed_instance, generic_args);
    let super_trait = mono_args[0].expect_ty();
    let target_trait = mono_args[1].expect_ty();

    // Compute outlives entries in transport/origin walk-position space by
    // composing through the call_id chain.
    let callee_instance = Instance::expect_resolve(
        tcx,
        ty::TypingEnv::fully_monomorphized(),
        def_id,
        mono_args,
        DUMMY_SP,
    );
    let mut call_site_outlives =
        tcx.augmented_outlives_for_call((delayed_instance, call_id, callee_instance));
    let root_transport_slots = region_slots_of_ty(super_trait);
    let total_transport_slots = root_transport_slots + region_slots_of_ty(target_trait);
    let origin_positions =
        compose_all_through_chain(tcx, delayed_instance, call_id, total_transport_slots);

    // Ground-level fallback: build outlives from borrowck region summary
    // when the augmented query returned nothing and the caller is not
    // already augmented. LocalOnly positions are naturally excluded —
    // they don't appear in the SCC-based outlives env, so
    // `identity_outlives_for_origin_positions` never emits them.
    if call_site_outlives.is_empty() && !delayed_instance.has_outlives_entries() {
        let origin_def_id = call_id[0].0;
        let origin_local_id = call_id[0].1;
        let summary = tcx.borrowck_region_summary(origin_def_id);
        if let Some(mapping) = summary.call_site_mappings.get(&origin_local_id) {
            call_site_outlives =
                identity_outlives_for_origin_positions(tcx, summary, mapping, &origin_positions);
        }
    }

    // Intern origin_positions for the query cache key.
    let interned_origins: &'tcx [Option<usize>] =
        tcx.arena.alloc_from_iter(origin_positions.iter().copied());

    let safe = tcx.is_lifetime_erasure_safe((
        super_trait,
        target_trait,
        interned_origins,
        call_site_outlives,
    ));
    Some(build_erasure_safe_rvalue(tcx, safe))
}

// ── Body patching ───────────────────────────────────────────────────────────

/// Walk a MIR body's terminators and replace trait-cast intrinsic calls with
/// constant assignments.
///
/// For each `Call` terminator whose callee matches a trait-cast intrinsic:
/// 1. The resolved constant `Rvalue` is computed (via
///    [`resolve_table_callee`] for table-dependent intrinsics, or
///    [`resolve_erasure_safe_callee`] for the erasure-safe intrinsic).
/// 2. The call is replaced with an `Assign` statement writing the constant
///    to the call's `destination`, followed by a `Goto` to the original
///    return target.
///
/// This preserves unwind behavior trivially since the resolved assignment
/// cannot panic.
pub(crate) fn patch_intrinsic_calls<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    delayed_instance: Instance<'tcx>,
    resolutions: &IntrinsicResolutions<'tcx>,
) {
    // Two-pass: first collect patches (read-only), then apply them (mutate).
    let mut patches: Vec<(BasicBlock, Place<'tcx>, Option<BasicBlock>, Rvalue<'tcx>)> = Vec::new();

    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        if let TerminatorKind::Call { ref func, ref call_id, destination, target, .. } =
            bb_data.terminator().kind
        {
            let resolved = resolve_table_callee(tcx, resolutions, delayed_instance, call_id, func)
                .or_else(|| resolve_erasure_safe_callee(tcx, delayed_instance, call_id, func));
            if let Some(rvalue) = resolved {
                patches.push((bb, destination, target, rvalue));
            }
        }
    }

    for (bb, destination, target, rvalue) in patches {
        let bb_data = &mut body.basic_blocks_mut()[bb];
        let target_bb = target.expect("trait-cast intrinsic calls always have a return target");
        let source_info = bb_data.terminator().source_info;

        // Replace the call with: _dest = <constant>; goto -> target;
        bb_data.statements.push(Statement::new(
            source_info,
            StatementKind::Assign(Box::new((destination, rvalue))),
        ));
        bb_data.terminator_mut().kind = TerminatorKind::Goto { target: target_bb };
    }
}
