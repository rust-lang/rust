/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::collections::{HashMap, HashSet};

use melior::dialect::scf;
use melior::ir::attribute::IntegerAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::IntegerType;
use melior::ir::{
    Attribute, Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, TypeLike,
    Value, ValueLike,
};
use melior::utility::register_all_llvm_translations;
use rustc_abi::{FieldIdx, FieldsShape, Size as AbiSize};
use rustc_ast::{IntTy, MutTy, UintTy};
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::{CtfeProvenance, GlobalAlloc, Scalar, alloc_range};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, BinOp, Body, CastKind, Const, ConstOperand,
    ConstValue, Local, NonDivergingIntrinsic, Operand, Place, ProjectionElem, Rvalue, Statement,
    StatementKind, UnevaluatedConst as MirUnevaluatedConst,
};
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{
    self, AdtDef, ConstKind, EarlyBinder, GenericArg, Instance, ParamConst, ScalarInt, Ty, TyCtxt,
    TyKind, TypingEnv, adjustment::PointerCoercion,
};
use rustc_mlir::load_all_dialects;
use rustc_mlir::shared::arith::{Int, Predicate, create_constant, create_int_constant};
use rustc_mlir::shared::attr::create_scalar_attr;
use rustc_mlir::shared::builtin::{tensor_type, tensor_type_like};
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tensor::splat;
use rustc_mlir::triton::{create_func, int_to_ptr, load_triton_dialect, pointer_type};
use rustc_span::DUMMY_SP;

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::types::TypeMapper;
use crate::mlir::errors::MlirError;

mod location;
mod ops;
mod types;

use types::is_option_ty;

type SsaValues<'c, 'p> = HashMap<Local, Value<'c, 'p>>;

/// Tracks `Option<T>` MIR locals that must not be materialised as MLIR SSA values.
/// `None` entry  → the local holds the `None` variant (no inner value).
/// `Some(v)` entry → the local holds `Some(v)` with this MLIR value as the inner.
/// These locals are NEVER inserted into `SsaValues`.
type OptionTable<'c, 'p> = HashMap<Local, Option<Value<'c, 'p>>>;

/// Tracks tuple MIR locals (including the return place `_0` for tuple-returning functions).
/// The vec contains the individual MLIR values for each tuple field in order.
type TupleTable<'c, 'p> = HashMap<Local, Vec<Value<'c, 'p>>>;

/// Constant integer arrays (e.g. shape arrays like `[BLOCK_SIZE]`).
/// Keyed by the MIR `Local` that holds the array; value is the elements as `i64`.
type ConstArrays = HashMap<Local, Vec<i64>>;

/// Maps a raw-pointer local (from `&raw const arr`) to the `ConstArrays` key it was derived from.
type PtrToConstArray = HashMap<Local, Local>;

/// Maps a MIR local that holds `&[i32]` (built from a const array via `slice_from_raw_parts` /
/// `transmute`) to the shape extracted from the underlying `ConstArrays` entry.
type SliceShape = HashMap<Local, Vec<i64>>;

/// Runtime (non-constant) arrays: elements are SSA `Value`s instead of `i64` literals.
type DynArrays<'c, 'p> = HashMap<Local, Vec<Value<'c, 'p>>>;

/// Maps a raw-pointer local derived from a `dyn_array` to the source array local.
type PtrToDynArray = HashMap<Local, Local>;

/// Slice (fat pointer) whose elements are runtime SSA values — parallel to `SliceShape` for
/// the dynamic case.
type SliceDynValues<'c, 'p> = HashMap<Local, Vec<Value<'c, 'p>>>;

/// Maps a descriptor local to its block shape (the tile dimensions it loads).
type DescBlockShapes = HashMap<Local, Vec<i64>>;

/// Maps a MIR local to its statically-known discriminant value (u64).
/// Built by pre-scanning for `SetDiscriminant` and `Discriminant` assignments
/// so that SwitchInt on Option<T> locals can be constant-folded during codegen.
type ConstDiscLocals = HashMap<Local, u64>;

/// Codegen state threaded through all statement/terminator handlers.
pub(crate) struct CodegenState<'c, 'p> {
    pub(crate) ssa_values: SsaValues<'c, 'p>,
    pub(crate) option_table: OptionTable<'c, 'p>,
    pub(crate) tuple_fields: TupleTable<'c, 'p>,
    /// Constant integer arrays derived from aggregate literals.
    pub(crate) const_arrays: ConstArrays,
    /// `local → array_local`: a raw pointer derived from a const-array alloca.
    pub(crate) ptr_to_const_array: PtrToConstArray,
    /// `local → shape`: a fat-pointer slice `&[i32]` whose shape is statically known.
    pub(crate) slice_shape: SliceShape,
    /// Runtime arrays whose elements are MLIR SSA values.
    pub(crate) dyn_arrays: DynArrays<'c, 'p>,
    /// `ptr_local → array_local`: pointer derived from a `dyn_arrays` entry.
    pub(crate) ptr_to_dyn_array: PtrToDynArray,
    /// Fat-pointer slices whose elements are runtime MLIR values.
    pub(crate) slice_dyn_values: SliceDynValues<'c, 'p>,
    /// Block shapes for tensor descriptors: `desc_local → [dim0, dim1, ...]`.
    pub(crate) desc_block_shapes: DescBlockShapes,
    /// When generating inside a `scf.for` body: the MIR basic block that is the loop header.
    /// `codegen_goto` checks this and emits `scf.yield` instead of `cf.br` on the back edge.
    pub(crate) loop_header_bb: Option<BasicBlock>,
    /// The MIR locals whose values must be yielded at the back edge of the active `scf.for`.
    pub(crate) loop_iter_carry_locals: Vec<Local>,
    /// MIR body blocks that are merged into the single scf.for body MLIR block.
    /// `codegen_goto` skips `cf.br` when the target is in this set (within-body jumps).
    pub(crate) loop_body_bbs: HashSet<BasicBlock>,
    /// Statically-known discriminant values for Option<T> locals (from SetDiscriminant).
    /// Used by `codegen_switch_int` to constant-fold branches on Option discriminants.
    pub(crate) const_disc_locals: ConstDiscLocals,
    /// Join blocks (2+ predecessors) → list of MIR locals that need phi block args there.
    pub(crate) phi_join_locals: HashMap<BasicBlock, Vec<Local>>,
    /// (join_block, local) → the MLIR block argument value representing the phi node.
    pub(crate) phi_block_args: HashMap<(BasicBlock, Local), Value<'c, 'p>>,
    /// Saves the ssa_values for phi locals just before the join block is processed.
    /// Later predecessors (processed after the join in DFS) use these saved values
    /// instead of the stale join-block arg values that are now in ssa_values.
    pub(crate) pre_join_ssa_values: HashMap<(BasicBlock, Local), Value<'c, 'p>>,
}

impl<'c, 'p> CodegenState<'c, 'p> {
    fn new() -> Self {
        Self {
            ssa_values: HashMap::new(),
            option_table: HashMap::new(),
            tuple_fields: HashMap::new(),
            const_arrays: HashMap::new(),
            ptr_to_const_array: HashMap::new(),
            slice_shape: HashMap::new(),
            dyn_arrays: HashMap::new(),
            ptr_to_dyn_array: HashMap::new(),
            slice_dyn_values: HashMap::new(),
            desc_block_shapes: HashMap::new(),
            loop_header_bb: None,
            loop_iter_carry_locals: Vec::new(),
            loop_body_bbs: HashSet::new(),
            const_disc_locals: HashMap::new(),
            phi_join_locals: HashMap::new(),
            phi_block_args: HashMap::new(),
            pre_join_ssa_values: HashMap::new(),
        }
    }
}

/// Information about a detected Range-based `for` loop in the MIR.
#[derive(Debug)]
struct RangeLoopInfo {
    /// The loop header block (checks the Lt condition and branches, OR calls Range::next).
    header_bb: BasicBlock,
    /// First block of the loop body (taken when the condition is true / Some).
    body_entry_bb: BasicBlock,
    /// All loop body blocks in execution order (body_entry through back_edge).
    body_bbs: Vec<BasicBlock>,
    /// The block whose `Goto` back-edge targets the header.
    back_edge_bb: BasicBlock,
    /// The block taken when the loop exits.
    exit_bb: BasicBlock,
    /// The Range.start local that is incremented each iteration.
    counter_local: Local,
    /// The loop upper-bound local (Range.end / k_tiles). None when bound is a constant.
    bound_local: Option<Local>,
    /// The loop upper-bound as a compile-time constant. None when bound is a runtime local.
    bound_const: Option<i64>,
    /// The "k" induction local (pre-increment counter, extracted from Some).
    induction_local: Local,
    /// Locals that carry values across iterations (loop-carried values).
    iter_carry_locals: Vec<Local>,
    /// For Call-header pattern (Rust 1.93+): the SwitchInt block after Range::next.
    switch_bb: Option<BasicBlock>,
    /// For Call-header pattern: the destination local of the Range::next call (Option<i32>).
    next_result_local: Option<Local>,
}

/// Pre-scan all MIR blocks to collect statically-known discriminant values.
///
/// Handles two patterns:
/// 1. `SetDiscriminant { place, variant_index }` → `place.local` has variant_index as discriminant
/// 2. `_d = Discriminant(_opt)` where `_opt` already has a known discriminant → `_d` inherits it
/// 3. `_local = const VALUE` where VALUE is a scalar constant (e.g. `_149 = const USE_BIAS`)
///
/// Constants are instantiated via `instance` so that const generic parameters (e.g. `USE_BIAS`)
/// get their concrete values (e.g. `false`) after monomorphization.
fn compute_const_disc_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
) -> HashMap<Local, u64> {
    use rustc_middle::mir::interpret::Scalar;
    let mut result: HashMap<Local, u64> = HashMap::new();

    let extract_scalar_const = |c: &ConstOperand<'tcx>| -> Option<u64> {
        // Instantiate the constant (substitutes generic params like USE_BIAS → false).
        let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(c.const_),
        );
        match instantiated {
            Const::Val(ConstValue::Scalar(Scalar::Int(s)), _) => {
                Some(s.to_bits_unchecked() as u64)
            }
            Const::Ty(_, ref ck) => {
                if let ConstKind::Value(cv) = ck.kind() {
                    cv.valtree.try_to_leaf().map(|s| s.to_bits_unchecked() as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    // Pass 1: collect SetDiscriminant statements and plain scalar const assignments.
    for (_, bb_data) in mir.basic_blocks.iter_enumerated() {
        for stmt in &bb_data.statements {
            match &stmt.kind {
                StatementKind::SetDiscriminant { place, variant_index } => {
                    if place.projection.is_empty() {
                        result.insert(place.local, variant_index.as_u32() as u64);
                    }
                }
                StatementKind::Assign(assign) => {
                    let (place, rvalue) = assign.as_ref();
                    if place.projection.is_empty() {
                        if let Rvalue::Use(Operand::Constant(c)) = rvalue {
                            if let Some(v) = extract_scalar_const(c) {
                                result.insert(place.local, v);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Pass 2: propagate through Discriminant-taking and copy assignments.
    let mut changed = true;
    while changed {
        changed = false;
        for (_, bb_data) in mir.basic_blocks.iter_enumerated() {
            for stmt in &bb_data.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, rvalue) = assign.as_ref();
                    if place.projection.is_empty() {
                        let known = match rvalue {
                            Rvalue::Discriminant(src) => result.get(&src.local).copied(),
                            Rvalue::Use(Operand::Copy(p) | Operand::Move(p))
                                if p.projection.is_empty() =>
                            {
                                result.get(&p.local).copied()
                            }
                            _ => None,
                        };
                        if let Some(v) = known {
                            if result.insert(place.local, v).is_none() {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

/// Extract a constant integer value from a SwitchInt discriminant operand, if statically known.
///
/// Handles:
/// 1. `Operand::Constant` with a scalar integer value (e.g. `BIAS = false`)
/// 2. `Operand::Copy/Move` of a local in `const_disc_locals` (from SetDiscriminant / Discriminant)
pub(crate) fn extract_switch_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    discr: &Operand<'tcx>,
    const_disc_locals: &HashMap<Local, u64>,
) -> Option<u64> {
    use rustc_middle::mir::interpret::Scalar;
    let result = match discr {
        Operand::Constant(c) => {
            // Instantiate generic const params (e.g. USE_BIAS → false) before extracting value.
            let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(c.const_),
            );
            match instantiated {
                Const::Val(ConstValue::Scalar(Scalar::Int(s)), _) => {
                    Some(s.to_bits_unchecked() as u64)
                }
                Const::Ty(_, ref ck) => {
                    if let ConstKind::Value(cv) = ck.kind() {
                        cv.valtree.try_to_leaf().map(|s| s.to_bits_unchecked() as u64)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Operand::Copy(p) | Operand::Move(p) if p.projection.is_empty() => {
            const_disc_locals.get(&p.local).copied()
        }
        _ => None,
    };
    println!("[SWITCH-CONST] discr={:?} → {:?}", discr, result);
    result
}

/// BFS reachability analysis from bb0, constant-folding `SwitchInt` when the discriminant
/// is statically known.  Returns (set, vec) where the vec preserves BFS order — the correct
/// topological order for SSA value availability.  Use the vec for block *processing* order
/// and the set for O(1) membership tests.
fn compute_reachable_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
    const_disc_locals: &HashMap<Local, u64>,
) -> (HashSet<BasicBlock>, Vec<BasicBlock>) {
    use rustc_middle::mir::TerminatorKind;

    println!("[REACH-START] total blocks={}", mir.basic_blocks.len());
    let mut reachable: HashSet<BasicBlock> = HashSet::new();
    let mut ordered: Vec<BasicBlock> = Vec::new();
    let mut queue: Vec<BasicBlock> = vec![BasicBlock::from_u32(0)];

    while let Some(bb) = queue.pop() {
        if !reachable.insert(bb) {
            continue;
        }
        ordered.push(bb);
        let bb_data = &mir.basic_blocks[bb];
        println!("[REACH-TERM] bb={:?} kind={}", bb, match &bb_data.terminator().kind {
            TerminatorKind::Goto { .. } => "Goto",
            TerminatorKind::SwitchInt { .. } => "SwitchInt",
            TerminatorKind::Return => "Return",
            TerminatorKind::Unreachable => "Unreachable",
            TerminatorKind::Call { .. } => "Call",
            TerminatorKind::Drop { .. } => "Drop",
            TerminatorKind::Assert { .. } => "Assert",
            _ => "Other",
        });
        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } => {
                queue.push(*target);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                println!("[REACH-BFS] SwitchInt at {:?}: discr={:?}", bb, discr);
                // Constant-fold when the discriminant is statically known.
                if let Some(const_val) = extract_switch_const(tcx, instance, discr, const_disc_locals) {
                    let target = targets
                        .iter()
                        .find(|(val, _)| *val == const_val as u128)
                        .map(|(_, bb)| bb)
                        .unwrap_or_else(|| targets.otherwise());
                    println!(
                        "[REACH] constant-folding SwitchInt: discr const={} → {:?}",
                        const_val, target
                    );
                    queue.push(target);
                    continue;
                }
                // Non-constant: all successors are reachable.
                for (_, target) in targets.iter() {
                    queue.push(target);
                }
                queue.push(targets.otherwise());
            }
            TerminatorKind::Return | TerminatorKind::Unreachable => {}
            TerminatorKind::Call { target, .. } => {
                if let Some(t) = target {
                    queue.push(*t);
                }
            }
            TerminatorKind::Drop { target, .. } => {
                queue.push(*target);
            }
            TerminatorKind::Assert { target, .. } => {
                queue.push(*target);
            }
            _ => {
                for succ in bb_data.terminator().successors() {
                    queue.push(succ);
                }
            }
        }
    }

    (reachable, ordered)
}

/// For each join block (2+ reachable non-loop predecessors), collect the set of MIR locals
/// that are assigned via simple `Assign` statements in any direct predecessor block.
/// These locals need MLIR block arguments (phi nodes) at the join block so that the SSA
/// value used after the join is dominated by its definition regardless of the taken path.
///
/// Locals tracked as Option or tuple are excluded — those are managed via separate tables.
///
/// This function traces backward through single-predecessor linear chains from each
/// predecessor of the join block, so it catches assignments that are a few hops back.
fn compute_phi_join_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir: &Body<'tcx>,
    reachable_bbs: &HashSet<BasicBlock>,
    loop_region_blocks: &HashSet<BasicBlock>,
) -> HashMap<BasicBlock, Vec<Local>> {
    // Build a predecessor map restricted to non-loop reachable blocks.
    let mut pred_map: HashMap<BasicBlock, HashSet<BasicBlock>> = HashMap::new();
    for &bb in reachable_bbs {
        if loop_region_blocks.contains(&bb) {
            continue;
        }
        for succ in mir.basic_blocks[bb].terminator().successors() {
            if reachable_bbs.contains(&succ) && !loop_region_blocks.contains(&succ) {
                pred_map.entry(succ).or_default().insert(bb);
            }
        }
    }

    // Helper: collect all locals assigned in the linear chain starting at `start`,
    // tracing backward through single-predecessor blocks.  Stops when we reach:
    //   • a block that is also a direct predecessor of the join block (the divergence pt)
    //   • a block with multiple forward successors (another branch point)
    //   • a block with multiple predecessors (another join point)
    //   • an already-visited block
    let collect_chain_locals = |start: BasicBlock, join_preds: &HashSet<BasicBlock>| -> HashSet<Local> {
        let mut locals: HashSet<Local> = HashSet::new();
        let mut current = start;
        let mut visited: HashSet<BasicBlock> = HashSet::new();
        loop {
            if !visited.insert(current) {
                break;
            }
            // Collect all assignments in this block.
            for stmt in &mir.basic_blocks[current].statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, _) = assign.as_ref();
                    if place.projection.is_empty() {
                        locals.insert(place.local);
                    }
                }
            }
            // Count forward successors (in the filtered graph).
            let succ_count = mir.basic_blocks[current]
                .terminator()
                .successors()
                .filter(|&s| reachable_bbs.contains(&s) && !loop_region_blocks.contains(&s))
                .count();
            if succ_count != 1 {
                // Divergence point — stop.
                break;
            }
            // Follow the single backward predecessor.
            let filtered_preds: Vec<BasicBlock> = pred_map
                .get(&current)
                .map(|ps| {
                    ps.iter()
                        .filter(|&&b| {
                            reachable_bbs.contains(&b) && !loop_region_blocks.contains(&b)
                        })
                        .copied()
                        .collect()
                })
                .unwrap_or_default();
            if filtered_preds.len() != 1 {
                break; // Another join point or entry — stop.
            }
            let prev = filtered_preds[0];
            // Stop when we reach another direct predecessor of the join block
            // (that is the divergence point shared by all paths).
            if join_preds.contains(&prev) {
                break;
            }
            current = prev;
        }
        locals
    };

    let mut result: HashMap<BasicBlock, Vec<Local>> = HashMap::new();
    for (join_bb, preds) in &pred_map {
        if preds.len() < 2 {
            continue;
        }
        // Count how many predecessor chains each local appears in.
        // A local is only a valid phi if it's redefined on 2+ distinct paths — otherwise
        // it has no value on the "other" path and can't be passed as a block arg.
        let mut chain_count: HashMap<Local, usize> = HashMap::new();
        let mut local_order: Vec<Local> = Vec::new();
        for &pred_bb in preds {
            for local in collect_chain_locals(pred_bb, preds) {
                let entry = chain_count.entry(local).or_insert(0);
                if *entry == 0 {
                    local_order.push(local);
                }
                *entry += 1;
            }
        }
        let mut phi_locals: Vec<Local> = Vec::new();
        for local in local_order {
            if chain_count[&local] < 2 {
                continue; // Only redefined on one path — no phi needed.
            }
            let raw_ty = mir.local_decls[local].ty;
            // Skip Option and Tuple locals — tracked in separate tables.
            if is_option_ty(tcx, raw_ty) {
                continue;
            }
            if matches!(raw_ty.kind(), rustc_middle::ty::TyKind::Tuple(_)) {
                continue;
            }
            phi_locals.push(local);
        }
        if !phi_locals.is_empty() {
            println!(
                "[PHI] join block {:?} has {} phi locals: {:?}",
                join_bb,
                phi_locals.len(),
                phi_locals
            );
            result.insert(*join_bb, phi_locals);
        }
    }
    result
}

/// Detect all Range-based `for` loops in the MIR body.
fn detect_range_loop<'tcx>(mir: &Body<'tcx>) -> Vec<RangeLoopInfo> {
    use rustc_middle::mir::TerminatorKind;

    let mut loops = Vec::new();
    for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
        if let TerminatorKind::Goto { target } = &bb_data.terminator().kind {
            if target.index() < bb.index() {
                // Back edge: bb → target (target is the loop header)
                let header_bb = *target;
                let back_edge_bb = bb;
                if let Some(info) = try_build_range_loop_info(mir, header_bb, back_edge_bb) {
                    loops.push(info);
                }
            }
        }
    }
    loops
}

fn try_build_range_loop_info<'tcx>(
    mir: &Body<'tcx>,
    header_bb: BasicBlock,
    back_edge_bb: BasicBlock,
) -> Option<RangeLoopInfo> {
    use rustc_middle::mir::TerminatorKind;

    let header_data = &mir.basic_blocks[header_bb];

    struct HeaderInfo {
        body_entry_bb: BasicBlock,
        exit_bb: BasicBlock,
        counter_local_hint: Option<Local>,
        bound_local: Option<Local>,
        bound_const: Option<i64>,
        switch_bb: Option<BasicBlock>,
        next_result_local: Option<Local>,
    }

    let hinfo: HeaderInfo = 'detect: {
        // Pattern 1: SwitchInt header with explicit Lt comparison (older Rust MIR).
        if let TerminatorKind::SwitchInt { discr, targets } = &header_data.terminator().kind {
            let cases: Vec<(u128, BasicBlock)> = targets.iter().collect();
            println!("[LOOP-DETECT] header={:?} back_edge={:?} SwitchInt cases={:?} otherwise={:?} discr={:?}",
                header_bb, back_edge_bb, cases, targets.otherwise(), discr);
            if cases.len() == 1 && cases[0].0 == 0 {
                let exit_bb = cases[0].1;
                let body_entry_bb = targets.otherwise();
                if let Operand::Move(p) | Operand::Copy(p) = discr {
                    if p.projection.is_empty() {
                        let discr_local = p.local;
                        if let Some((counter_local, bl)) =
                            find_lt_locals_in_stmts(&header_data.statements, discr_local)
                        {
                            println!("[LOOP-DETECT] Pattern1 detected: counter={:?} bound_local={:?}", counter_local, bl);
                            break 'detect HeaderInfo {
                                body_entry_bb,
                                exit_bb,
                                counter_local_hint: Some(counter_local),
                                bound_local: Some(bl),
                                bound_const: None,
                                switch_bb: None,
                                next_result_local: None,
                            };
                        }
                    }
                }
            }
            println!("[LOOP-DETECT] FAIL Pattern1: cases={:?}", cases);
        }

        // Pattern 2: Call header (Range::next) followed by SwitchInt on discriminant (Rust 1.93+).
        if let TerminatorKind::Call { target: Some(switch_bb_cand), destination, .. } =
            &header_data.terminator().kind
        {
            let switch_data = &mir.basic_blocks[*switch_bb_cand];
            if let TerminatorKind::SwitchInt { targets, .. } = &switch_data.terminator().kind {
                let cases: Vec<(u128, BasicBlock)> = targets.iter().collect();
                println!("[LOOP-DETECT] header={:?} back_edge={:?} Call+SwitchInt cases={:?} otherwise={:?}",
                    header_bb, back_edge_bb, cases, targets.otherwise());
                // Discriminant 0 = None (exit), discriminant 1 = Some (body)
                if cases.len() == 2 && cases[0].0 == 0 && cases[1].0 == 1
                    && destination.projection.is_empty()
                {
                    let exit_bb = cases[0].1;
                    let body_entry_bb = cases[1].1;
                    let next_result_local = destination.local;
                    if let Some((bc, bl)) = find_range_bound(mir, header_bb) {
                        println!("[LOOP-DETECT] Pattern2 detected: bound_const={:?} bound_local={:?}", bc, bl);
                        break 'detect HeaderInfo {
                            body_entry_bb,
                            exit_bb,
                            counter_local_hint: None,
                            bound_local: bl,
                            bound_const: bc,
                            switch_bb: Some(*switch_bb_cand),
                            next_result_local: Some(next_result_local),
                        };
                    }
                }
            }
            println!("[LOOP-DETECT] FAIL Pattern2: switch_bb={:?}", switch_bb_cand);
        }

        println!("[LOOP-DETECT] header={:?} back_edge={:?}: no pattern matched", header_bb, back_edge_bb);
        return None;
    };

    // Collect body blocks in topological (execution) order
    let body_bbs = collect_body_blocks_ordered(mir, hinfo.body_entry_bb, header_bb);
    println!("[LOOP-DETECT] body_bbs={:?} back_edge_in_body={}", body_bbs, body_bbs.contains(&back_edge_bb));
    if !body_bbs.contains(&back_edge_bb) {
        return None;
    }

    // Find the induction local: extracted from Option::Some via downcast projection
    let induction_result = find_induction_local_in_bbs(mir, &body_bbs);
    println!("[LOOP-DETECT] induction_local={:?}", induction_result);
    let induction_local = induction_result?;

    let counter_local = hinfo.counter_local_hint.unwrap_or(induction_local);

    // Find loop-carried locals (used before assigned in topological order).
    let mut iter_carry_locals = find_iter_carry_locals(mir, &body_bbs);
    println!("[LOOP-DETECT] iter_carry_locals (before filter)={:?}", iter_carry_locals);

    // For the Call-header pattern, exclude the Range::next result local (next_result_local)
    // from iter-carry: it is synthesised by scf.for, not a user-defined loop-carried value.
    if let Some(nrl) = hinfo.next_result_local {
        iter_carry_locals.retain(|&l| l != nrl);
    }
    println!("[LOOP-DETECT] iter_carry_locals (final)={:?}", iter_carry_locals);

    Some(RangeLoopInfo {
        header_bb,
        body_entry_bb: hinfo.body_entry_bb,
        body_bbs,
        back_edge_bb,
        exit_bb: hinfo.exit_bb,
        counter_local,
        bound_local: hinfo.bound_local,
        bound_const: hinfo.bound_const,
        induction_local,
        iter_carry_locals,
        switch_bb: hinfo.switch_bb,
        next_result_local: hinfo.next_result_local,
    })
}

/// Extract the loop upper bound from the Range aggregate that feeds the iterator in the
/// blocks preceding `header_bb`. Returns `(bound_const, bound_local)` where exactly one
/// of the two `Option`s is `Some`.
fn find_range_bound<'tcx>(
    mir: &Body<'tcx>,
    header_bb: BasicBlock,
) -> Option<(Option<i64>, Option<Local>)> {
    use rustc_middle::mir::StatementKind;
    for bb_idx in 0..header_bb.index() {
        let bb = BasicBlock::from_usize(bb_idx);
        for stmt in &mir.basic_blocks[bb].statements {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (_, rvalue) = assign.as_ref();
                if let Rvalue::Aggregate(kind, fields) = rvalue
                    && matches!(kind.as_ref(), AggregateKind::Adt(..))
                {
                    if fields.len() == 2 {
                        // Treat [start, end] → look at index 1 (end).
                        let end_op = &fields[FieldIdx::from_usize(1)];
                        match end_op {
                            Operand::Constant(c) => {
                                if let Some(v) = try_const_operand_as_i64(c) {
                                    return Some((Some(v), None));
                                }
                            }
                            Operand::Copy(p) | Operand::Move(p)
                                if p.projection.is_empty() =>
                            {
                                return Some((None, Some(p.local)));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Try to extract a constant integer value from a `ConstOperand` without needing tcx/instance.
/// Works for monomorphized scalar constants (`Const::Val`) and simple `Const::Ty` value-tree consts.
fn try_const_operand_as_i64(c: &ConstOperand<'_>) -> Option<i64> {
    c.const_.try_to_scalar_int().map(|s| s.to_bits_unchecked() as i64)
}

/// Find the Lt statement that assigns `discr_local` and extract (counter_local, bound_local).
fn find_lt_locals_in_stmts<'tcx>(
    stmts: &[Statement<'tcx>],
    discr_local: Local,
) -> Option<(Local, Local)> {
    for stmt in stmts {
        if let StatementKind::Assign(assign) = &stmt.kind {
            let (dest, rvalue) = assign.as_ref();
            if dest.local == discr_local && dest.projection.is_empty() {
                if let Rvalue::BinaryOp(BinOp::Lt, operands) = rvalue {
                    let (lhs, rhs) = operands.as_ref();
                    let counter_copy = match lhs {
                        Operand::Move(p) | Operand::Copy(p) if p.projection.is_empty() => p.local,
                        _ => return None,
                    };
                    let bound_local = match rhs {
                        Operand::Move(p) | Operand::Copy(p) if p.projection.is_empty() => p.local,
                        _ => return None,
                    };
                    // Resolve the counter_copy (might be a copy of the real counter)
                    let counter_local = find_copy_source_in_stmts(stmts, counter_copy)
                        .unwrap_or(counter_copy);
                    return Some((counter_local, bound_local));
                }
            }
        }
    }
    None
}

/// Find the source local if `local` is assigned `copy src` in the statements.
fn find_copy_source_in_stmts<'tcx>(stmts: &[Statement<'tcx>], local: Local) -> Option<Local> {
    for stmt in stmts.iter().rev() {
        if let StatementKind::Assign(assign) = &stmt.kind {
            let (dest, rvalue) = assign.as_ref();
            if dest.local == local && dest.projection.is_empty() {
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) = rvalue {
                    if src.projection.is_empty() {
                        return Some(src.local);
                    }
                }
            }
        }
    }
    None
}

/// Collect body blocks in execution order (from body_entry_bb, stopping before header_bb).
fn collect_body_blocks_ordered<'tcx>(
    mir: &Body<'tcx>,
    body_entry_bb: BasicBlock,
    header_bb: BasicBlock,
) -> Vec<BasicBlock> {
    use rustc_middle::mir::TerminatorKind;

    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = vec![body_entry_bb];

    while let Some(bb) = queue.first().copied() {
        queue.remove(0);
        if visited.contains(&bb) || bb == header_bb {
            continue;
        }
        visited.insert(bb);
        result.push(bb);

        let bb_data = &mir.basic_blocks[bb];
        match &bb_data.terminator().kind {
            TerminatorKind::Goto { target } if *target != header_bb => {
                queue.push(*target);
            }
            TerminatorKind::Call { target: Some(target), .. } if *target != header_bb => {
                queue.push(*target);
            }
            TerminatorKind::Assert { target, .. } if *target != header_bb => {
                queue.push(*target);
            }
            _ => {}
        }
    }
    result
}

/// Find the induction local: a local assigned via a downcast+field projection on an Option local.
fn find_induction_local_in_bbs<'tcx>(
    mir: &Body<'tcx>,
    body_bbs: &[BasicBlock],
) -> Option<Local> {
    for &bb in body_bbs {
        for stmt in &mir.basic_blocks[bb].statements {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (dest, rvalue) = assign.as_ref();
                if dest.projection.is_empty() {
                    if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) = rvalue {
                        // Downcast+field projection = Option::Some unwrap
                        let has_downcast = src.projection.iter().any(|p| {
                            matches!(p, ProjectionElem::Downcast(_, _))
                        });
                        if has_downcast {
                            return Some(dest.local);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Find loop-carried locals: call-destination locals that are used before their first assignment
/// in the topological order of the body blocks.
fn find_iter_carry_locals<'tcx>(mir: &Body<'tcx>, body_bbs: &[BasicBlock]) -> Vec<Local> {
    use rustc_middle::mir::TerminatorKind;

    // Collect all locals that are assigned anywhere in the loop body (statements or calls).
    let mut all_assigned: HashSet<Local> = HashSet::new();
    let mut ordered_candidates: Vec<Local> = Vec::new();
    for &bb in body_bbs {
        let bb_data = &mir.basic_blocks[bb];
        for stmt in &bb_data.statements {
            if let StatementKind::Assign(assign) = &stmt.kind {
                let (dest, _) = assign.as_ref();
                if dest.projection.is_empty() && all_assigned.insert(dest.local) {
                    ordered_candidates.push(dest.local);
                }
            }
        }
        if let TerminatorKind::Call { destination, .. } = &bb_data.terminator().kind {
            if destination.projection.is_empty() && all_assigned.insert(destination.local) {
                ordered_candidates.push(destination.local);
            }
        }
    }

    let mut iter_carry = Vec::new();

    for candidate in ordered_candidates {
        let mut assigned = false;
        let mut use_before_assign = false;

        'scan: for &bb in body_bbs {
            let bb_data = &mir.basic_blocks[bb];
            for stmt in &bb_data.statements {
                if !assigned && local_used_in_stmt(stmt, candidate) {
                    use_before_assign = true;
                    break 'scan;
                }
                if local_assigned_in_stmt(stmt, candidate) {
                    assigned = true;
                }
            }
            // Check terminator args (use before call assignment)
            if !assigned {
                if let TerminatorKind::Call { args, .. } = &bb_data.terminator().kind {
                    if args.iter().any(|a| operand_uses_local(&a.node, candidate)) {
                        use_before_assign = true;
                        break 'scan;
                    }
                }
            }
            // Call destination counts as assignment
            if let TerminatorKind::Call { destination, .. } = &bb_data.terminator().kind {
                if destination.local == candidate {
                    assigned = true;
                }
            }
        }

        if use_before_assign {
            iter_carry.push(candidate);
        }
    }
    iter_carry
}

fn local_used_in_stmt(stmt: &Statement<'_>, local: Local) -> bool {
    match &stmt.kind {
        StatementKind::Assign(assign) => {
            let (_, rvalue) = assign.as_ref();
            rvalue_uses_local(rvalue, local)
        }
        _ => false,
    }
}

fn local_assigned_in_stmt(stmt: &Statement<'_>, local: Local) -> bool {
    match &stmt.kind {
        StatementKind::Assign(assign) => {
            let (dest, _) = assign.as_ref();
            dest.local == local && dest.projection.is_empty()
        }
        _ => false,
    }
}

fn rvalue_uses_local(rvalue: &Rvalue<'_>, local: Local) -> bool {
    match rvalue {
        Rvalue::Use(op) | Rvalue::Repeat(op, _) => operand_uses_local(op, local),
        Rvalue::Cast(_, op, _) => operand_uses_local(op, local),
        Rvalue::BinaryOp(_, operands) => {
            let (l, r) = operands.as_ref();
            operand_uses_local(l, local) || operand_uses_local(r, local)
        }
        Rvalue::UnaryOp(_, op) => operand_uses_local(op, local),
        Rvalue::Aggregate(_, fields) => fields.iter().any(|f| operand_uses_local(f, local)),
        Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => place.local == local,
        _ => false,
    }
}

fn operand_uses_local(op: &Operand<'_>, local: Local) -> bool {
    match op {
        Operand::Copy(p) | Operand::Move(p) => p.local == local,
        Operand::Constant(_) | Operand::RuntimeChecks(_) => false,
    }
}

pub(crate) struct TritonCodegen<'a> {
    module: &'a MlirModule<'static>,
    type_mapper: TypeMapper,
}

impl<'a> TritonCodegen<'a> {
    pub fn new(module: &'a MlirModule<'static>) -> Self {
        let context = module.context();

        load_all_dialects(context);
        register_all_llvm_translations(context);
        load_triton_dialect(context);

        Self { module, type_mapper: TypeMapper::new() }
    }

    fn to_scalar_int<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        node: &Operand<'tcx>,
    ) -> Result<ScalarInt, MlirError> {
        match node {
            Operand::Constant(c) => {
                // We expect the constant to have a value that tells us the discriminant/variant
                match c.const_ {
                    Const::Val(ConstValue::Scalar(Scalar::Int(scalar_int)), _) => Ok(scalar_int),
                    Const::Ty(ty, const_val) => {
                        // Handle const generic parameters of integer type
                        if !ty.is_integral() {
                            return Err(MlirError::InvalidScalar { node: format!("{:?}", node) });
                        }
                        match const_val.kind() {
                            ConstKind::Param(param) => {
                                let value = instance.args.const_at(param.index as usize).to_value();
                                let scalar_int = value.try_to_leaf().ok_or_else(|| {
                                    MlirError::InvalidScalar { node: format!("{:?}", node) }
                                })?;
                                Ok(scalar_int)
                            }
                            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                        }
                    }
                    _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                }
            }
            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
        }
    }

    /// Try to extract i64 values from a promoted constant of type `&[i32; N]`.
    ///
    /// This handles the MIR pattern generated by `&[BLOCK_SIZE]` when coercing to `&[i32]`:
    ///   `_x = const promoted[N] as &[i32] (PointerCoercion(Unsize, Implicit))`
    /// The promoted constant is a reference to a static allocation containing the array elements.
    /// Returns `Some(vec_of_i64_values)` on success, `None` if the pattern doesn't match.
    fn try_read_array_ref_const<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        src_ty: Ty<'tcx>,
        c: &Const<'tcx>,
    ) -> Option<Vec<i64>> {
        // Source type must be &[i32; N] (reference to fixed-size integral array).
        let (elem_ty, len_const) = match src_ty.kind() {
            TyKind::Ref(_, inner, _) => match inner.kind() {
                TyKind::Array(et, lc) => (et, lc),
                _ => return None,
            },
            _ => return None,
        };
        if !elem_ty.is_integral() {
            return None;
        }
        let elem_size_bytes = match elem_ty.kind() {
            TyKind::Int(IntTy::I8) | TyKind::Uint(UintTy::U8) => 1u64,
            TyKind::Int(IntTy::I16) | TyKind::Uint(UintTy::U16) => 2,
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => 4,
            TyKind::Int(IntTy::I64) | TyKind::Uint(UintTy::U64) => 8,
            _ => return None,
        };
        let n = len_const.try_to_target_usize(tcx)? as usize;

        // Resolve the constant to a ConstValue.
        // Promoted constants from generic functions arrive as Const::Unevaluated with generic args;
        // we must substitute the instance's args then evaluate via const_eval_resolve.
        let const_val: ConstValue = match c {
            Const::Val(cv, _) => *cv,
            Const::Unevaluated(uv, _) => {
                // Substitute the function's generic params with the instance's concrete args.
                let concrete_args = EarlyBinder::bind(uv.args).instantiate(tcx, instance.args);
                let concrete_uv =
                    MirUnevaluatedConst { def: uv.def, args: concrete_args, promoted: uv.promoted };
                match tcx.const_eval_resolve(
                    TypingEnv::fully_monomorphized(),
                    concrete_uv,
                    DUMMY_SP,
                ) {
                    Ok(cv) => cv,
                    Err(_) => return None,
                }
            }
            _ => return None,
        };

        // The ConstValue for `&[T; N]` is a thin pointer (Scalar::Ptr) to the allocation.
        // Dereference it to get the allocation that holds the array bytes.
        let (alloc_id, base_offset) = match const_val {
            ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
                let (prov, offset): (CtfeProvenance, _) = ptr.prov_and_relative_offset();
                (prov.alloc_id(), offset)
            }
            ConstValue::Indirect { alloc_id, offset } => (alloc_id, offset),
            _ => return None,
        };

        let GlobalAlloc::Memory(const_alloc) = tcx.global_alloc(alloc_id) else { return None };
        let alloc = const_alloc.inner();

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let elem_offset = base_offset + AbiSize::from_bytes(i as u64 * elem_size_bytes);
            let range = alloc_range(elem_offset, AbiSize::from_bytes(elem_size_bytes));
            let scalar = alloc.read_scalar(&tcx, range, false).ok()?;
            let val = match scalar {
                Scalar::Int(s) => match elem_size_bytes {
                    1 => s.to_i8() as i64,
                    2 => s.to_i16() as i64,
                    4 => s.to_i32() as i64,
                    8 => s.to_i64(),
                    _ => return None,
                },
                _ => return None,
            };
            result.push(val);
        }
        Some(result)
    }

    /// The value is assumed to be a scalar of the same type as the tensor.
    /// The result is a tensor of the same shape as the provided tensor, with the scalar values repeated.
    fn like_tensor<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        tensor: Value<'a, 'a>,
        value: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let tensor_type = tensor_type_like(
            tensor
                .r#type()
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?,
            value.r#type(),
        )
        .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let splat_op: Operation<'_> =
            splat(&self.module.context(), location, value, tensor_type.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).unwrap();

        mlir_block.append_operation(splat_op);
        Ok(result.into())
    }

    /// Generate a `scf.for` loop for a detected Range-based loop, emitting the op into
    /// `init_mlir_block`. After the loop, control branches to `loop_info.exit_bb`.
    #[allow(clippy::too_many_arguments)]
    fn codegen_scf_for_loop<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        loop_info: &RangeLoopInfo,
        init_mlir_block: &BlockRef<'a, 'a>,
        func_op: &Operation<'a>,
        state: &mut CodegenState<'a, 'a>,
        outer_basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        location: Location<'a>,
    ) -> Result<(), MlirError> {
        let ctx = self.module.context();
        let i32_ty: melior::ir::Type<'_> = IntegerType::new(ctx, 32).into();

        // Emit lb = 0 and step = 1
        let lb_op: Operation<'a> =
            melior::dialect::arith::constant(ctx, IntegerAttribute::new(i32_ty, 0).into(), location)
                .into();
        let lb: Value<'a, 'a> = lb_op.result(0).unwrap().into();
        init_mlir_block.append_operation(lb_op);

        let step_op: Operation<'a> =
            melior::dialect::arith::constant(ctx, IntegerAttribute::new(i32_ty, 1).into(), location)
                .into();
        let step: Value<'a, 'a> = step_op.result(0).unwrap().into();
        init_mlir_block.append_operation(step_op);

        // Upper bound: either a compile-time constant or a runtime local.
        let ub: Value<'a, 'a> = if let Some(const_val) = loop_info.bound_const {
            let op: Operation<'a> = melior::dialect::arith::constant(
                ctx,
                IntegerAttribute::new(i32_ty, const_val).into(),
                location,
            )
            .into();
            let v: Value<'a, 'a> = op.result(0).unwrap().into();
            init_mlir_block.append_operation(op);
            v
        } else if let Some(bound_local) = loop_info.bound_local {
            *state.ssa_values.get(&bound_local).unwrap_or_else(|| {
                panic!("scf.for: bound local {:?} not in ssa_values", bound_local)
            })
        } else {
            panic!("RangeLoopInfo has neither bound_local nor bound_const")
        };

        // Iter-arg initial values and types
        let iter_arg_inits: Vec<Value<'a, 'a>> = loop_info
            .iter_carry_locals
            .iter()
            .map(|local| {
                *state.ssa_values.get(local).unwrap_or_else(|| {
                    panic!("scf.for: iter-carry local {:?} not in ssa_values", local)
                })
            })
            .collect();
        let iter_arg_types: Vec<melior::ir::Type<'a>> =
            iter_arg_inits.iter().map(|v| v.r#type()).collect();

        // Build the scf.for op with an empty body region; we fill the region afterwards.
        let for_op: Operation<'a> = OperationBuilder::new("scf.for", location)
            .add_operands(&[lb, ub, step])
            .add_operands(&iter_arg_inits)
            .add_results(&iter_arg_types)
            .add_regions([Region::new()])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;

        // Create a SINGLE body block for the entire scf.for body.
        // scf.for requires the body region to have exactly one block.
        let body_region = for_op.region(0).expect("scf.for must have a body region");

        // Build a combined block map: outer + loop body blocks.
        let mut combined_blocks: HashMap<BasicBlock, BlockRef<'a, 'a>> =
            outer_basic_blocks.clone();

        // The entry block gets: (induction_var: i32, iter_arg_0: T0, ...)
        let mut entry_block_args: Vec<(melior::ir::Type<'a>, Location<'a>)> =
            vec![(i32_ty, location)];
        for &ty in &iter_arg_types {
            entry_block_args.push((ty, location));
        }

        // One single block; all body MIR blocks map to it.
        let body_block = Block::new(&entry_block_args);
        let body_block_ref = body_region.append_block(body_block);
        for &body_bb in &loop_info.body_bbs {
            combined_blocks.insert(body_bb, body_block_ref);
        }

        // Map induction variable and iter-carry locals to body entry block arguments.
        {
            let entry_block_ref = *combined_blocks.get(&loop_info.body_entry_bb).unwrap();
            let iv: Value<'a, 'a> = entry_block_ref.argument(0).unwrap().into();
            // Both the "k" local and the counter local map to the scf.for IV.
            state.ssa_values.insert(loop_info.induction_local, iv);
            state.ssa_values.insert(loop_info.counter_local, iv);
            for (i, &local) in loop_info.iter_carry_locals.iter().enumerate() {
                let arg: Value<'a, 'a> = entry_block_ref.argument(i + 1).unwrap().into();
                state.ssa_values.insert(local, arg);
            }
            // For the Call-header pattern (Rust 1.93+): pre-populate the Range::next result local
            // as Some(iv) in the option_table so that `_induction = ((_next as Some).0)` resolves.
            if let Some(nrl) = loop_info.next_result_local {
                state.option_table.insert(nrl, Some(iv));
            }
        }

        // Set loop context so codegen_goto emits scf.yield at the back edge
        // and skips cf.br between body blocks (which share a single MLIR block).
        state.loop_header_bb = Some(loop_info.header_bb);
        state.loop_iter_carry_locals = loop_info.iter_carry_locals.clone();
        state.loop_body_bbs = loop_info.body_bbs.iter().copied().collect();

        // Process each body block in execution order.
        for &body_bb in &loop_info.body_bbs {
            let bb_data = &mir.basic_blocks[body_bb];
            self.codegen_basic_block(
                tcx,
                instance,
                mir,
                body_bb,
                bb_data,
                func_op,
                state,
                &combined_blocks,
            )?;
        }

        // Clear loop context.
        state.loop_header_bb = None;
        state.loop_iter_carry_locals.clear();
        state.loop_body_bbs.clear();

        // Collect scf.for results before moving the op.
        let for_results: Vec<Value<'a, 'a>> = (0..loop_info.iter_carry_locals.len())
            .map(|i| for_op.result(i).unwrap().into())
            .collect();

        // Append the scf.for to the init block.
        init_mlir_block.append_operation(for_op.into());

        // Map results back to iter-carry locals (post-loop values).
        for (i, &local) in loop_info.iter_carry_locals.iter().enumerate() {
            state.ssa_values.insert(local, for_results[i]);
        }

        // Branch to the exit block.
        let exit_block = *outer_basic_blocks.get(&loop_info.exit_bb).unwrap_or_else(|| {
            panic!("scf.for: exit block {:?} not in basic_blocks", loop_info.exit_bb)
        });
        let br_op = rustc_mlir::shared::cf::create_cf_br(ctx, location, &exit_block)
            .map_err(|e| MlirError::CreateOperation { err: e })?;
        init_mlir_block.append_operation(br_op.into());

        Ok(())
    }

    fn codegen_function<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        fn_ty: Ty<'tcx>,
        instance: &Instance<'tcx>,
    ) -> Result<(), MlirError> {
        let mut state = CodegenState::new();
        let mut basic_blocks: HashMap<BasicBlock, BlockRef> = HashMap::new();

        // Downcast to a FnSig
        let fn_sig = fn_ty.fn_sig(tcx);
        let fn_sig = fn_sig.skip_binder(); // Remove late-bound lifetimes

        // Extract a friendly function name, preferring unmangled if possible
        let func_name = tcx.symbol_name(*instance).name;

        // Try to demangle using the Rust symbol demangling crate if available.
        // Since in rustc we don't always bring in the rustc-demangle crate, but
        // the symbol_name should generally be readable for non-generic items.
        // Otherwise, fallback to `def_path_str` (should give a crate-relative path).
        let friendly_name = if func_name.starts_with("_R") {
            // Looks like a Rust-mangled symbol. Try to show a better name.
            tcx.def_path_str(instance.def_id())
        } else {
            func_name.to_string()
        };

        eprintln!(
            "[DEBUG] TritonCodegen codegen_function: function name: {} (raw symbol: {})",
            friendly_name, func_name
        );

        // Skip Triton intrinsic function bodies — calls to these are intercepted at call-sites
        // by the codegen dispatch table, so the actual body is never compiled for GPU execution.
        // Their signatures may contain types (like `&[T]`) that have no valid static MLIR form.
        if fn_sig.inputs().iter().any(|ty| {
            matches!(ty.kind(), TyKind::Ref(_, inner, _) if matches!(inner.kind(), TyKind::Slice(_)))
        }) {
            eprintln!(
                "[DEBUG] TritonCodegen codegen_function: skipping intrinsic stub (has &[T] param): {}",
                friendly_name
            );
            return Ok(());
        }

        // Arguments — Option<T> params are excluded from the MLIR signature; they
        // are tracked in the option_table as None (absent) by default.
        let inputs: Vec<Ty<'tcx>> = fn_sig.inputs().iter().copied().collect();
        let arg_types: Vec<_> = inputs
            .iter()
            .filter(|ty| !is_option_ty(tcx, **ty))
            .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, ty))
            .collect();

        // Result type — flatten top-level Rust tuples into multiple MLIR return types.
        let ret_type = fn_sig.output();
        let ret_types: Vec<_> = if ret_type.is_unit() {
            vec![]
        } else if let TyKind::Tuple(elem_tys) = ret_type.kind() {
            elem_tys
                .iter()
                .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                .collect()
        } else {
            match self.type_mapper.map_type(self.module.context(), &tcx, &ret_type).to_result() {
                Ok(t) => vec![t],
                Err(_) => vec![],
            }
        };

        // Skip functions whose MLIR signature contains dynamic tensor types (tensor<?x...>).
        // These are Triton intrinsic stubs whose call-sites are intercepted by the dispatch table;
        // their generic tensor parameters have no statically-known shape and cannot be verified.
        let has_dynamic_tensor = arg_types.iter().chain(ret_types.iter()).any(|ty| ty.is_tensor());
        if has_dynamic_tensor {
            eprintln!(
                "[DEBUG] TritonCodegen codegen_function: skipping stub (has tensor<> param): {}",
                friendly_name
            );
            return Ok(());
        }

        // DEBUG output: print argument and result types
        eprintln!("[DEBUG] TritonCodegen: instance function signature (argument types):");
        for (i, arg_ty) in arg_types.iter().enumerate() {
            eprintln!("    arg[{}]: {}", i, arg_ty);
        }
        eprintln!(
            "[DEBUG] TritonCodegen: instance function signature (return type): {:?}",
            ret_types
        );

        // Iterate over MIR basic blocks and codegen each one
        let visibility = if func_name.ends_with("entry_point") { "public" } else { "private" };

        let func_loc =
            span_to_location(self.module.context(), tcx, tcx.def_span(instance.def_id()));
        let func_op: Operation = create_func(
            self.module.context(),
            func_loc,
            func_name,
            visibility,
            &arg_types,
            &ret_types,
            16,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let mir = tcx.instance_mir(instance.def);
        let location = func_loc;

        if func_name.contains("program_id") {
            println!(
                "[DEBUG] TritonCodegen::codegen_function: func_name: {:?}, arg_types: {:?}",
                func_name, arg_types
            );
            //println!("[DEBUG] TritonCodegen::codegen_function: mir: {:?}", mir);
        }

        // Pre-scan for statically-known discriminant values (e.g. Option locals set to None
        // by const generics).  Used to constant-fold SwitchInt and prune dead blocks.
        let const_disc_locals = compute_const_disc_locals(tcx, instance, mir);
        state.const_disc_locals = const_disc_locals.clone();

        // BFS reachability analysis: only reachable blocks get MLIR blocks.
        // The returned vec preserves BFS order — use it for *processing* so that each
        // block is processed only after all blocks that dominate it (i.e., SSA values
        // from predecessor blocks are already in state.ssa_values).
        let (reachable_bbs, bfs_ordered_bbs) =
            compute_reachable_blocks(tcx, instance, mir, &const_disc_locals);
        println!(
            "[DEBUG] TritonCodegen: reachable_bbs: {:?} (total MIR blocks: {})",
            reachable_bbs,
            mir.basic_blocks.len()
        );

        // Detect all Range-based `for` loops. Loop body blocks live in the scf.for region, not
        // in the function region, so we skip creating MLIR blocks for them here.
        let loop_infos = detect_range_loop(mir);

        // The set of MIR blocks that belong to any scf.for region.
        let mut loop_region_blocks: HashSet<BasicBlock> = HashSet::new();
        for l in &loop_infos {
            loop_region_blocks.insert(l.header_bb);
            if let Some(switch_bb) = l.switch_bb {
                loop_region_blocks.insert(switch_bb);
            }
            for &b in &l.body_bbs {
                loop_region_blocks.insert(b);
            }
        }

        // Map from init_bb → index into loop_infos for each loop.
        // The "init block" ends with `goto → header_bb`; we intercept it to emit scf.for.
        let mut loop_init_bb_map: HashMap<BasicBlock, usize> = HashMap::new();
        for (loop_idx, l) in loop_infos.iter().enumerate() {
            if let Some(init_bb) = mir.basic_blocks.indices().find(|&bb| {
                if loop_region_blocks.contains(&bb) {
                    return false;
                }
                if bb >= l.header_bb {
                    return false;
                }
                matches!(
                    &mir.basic_blocks[bb].terminator().kind,
                    rustc_middle::mir::TerminatorKind::Goto { target } if *target == l.header_bb
                )
            }) {
                loop_init_bb_map.insert(init_bb, loop_idx);
            }
        }

        println!(
            "[DEBUG] TritonCodegen: loop_infos count={}, loop_init_bb_map: {:?}",
            loop_infos.len(), loop_init_bb_map
        );

        // Compute phi (block-argument) locals for join blocks in the non-loop CFG.
        let phi_join_locals =
            compute_phi_join_locals(tcx, mir, &reachable_bbs, &loop_region_blocks);
        state.phi_join_locals = phi_join_locals;

        // Create MLIR blocks for all non-loop-region, reachable MIR blocks.
        // Index order is fine for block *creation* — MLIR blocks just need to exist before
        // they are branched to; actual codegen (where SSA values must be available) uses BFS order.
        for bb in &bfs_ordered_bbs {
            let bb = *bb;
            if loop_region_blocks.contains(&bb) {
                continue; // Loop body blocks are created inside the scf.for region.
            }

            let block = Block::new(&[]);
            if bb.index() == 0 {
                // Add non-Option function arguments as block arguments to the entry block.
                // Option<T> params are pre-populated into option_table as None.
                let mut mlir_arg_idx = 0;
                for (param_idx, input_ty) in inputs.iter().enumerate() {
                    let local = Local::from_usize(param_idx + 1);
                    if is_option_ty(tcx, *input_ty) {
                        state.option_table.insert(local, None);
                    } else {
                        let value = block.add_argument(arg_types[mlir_arg_idx], location);
                        state.ssa_values.insert(local, value);
                        mlir_arg_idx += 1;
                    }
                }
            }

            // Phi block args for join blocks are created lazily in codegen_goto /
            // codegen_switch_int so that tensor locals get their concrete shape type
            // (known only at codegen time) rather than the generic tensor<?xf32>.

            let block_ref =
                func_op.region(0).expect("tt.func must have a body region").append_block(block);
            basic_blocks.insert(bb, block_ref);
        }

        // Process all non-loop-region, reachable MIR blocks in BFS order so that SSA values
        // defined in a block are always available when successor blocks are processed,
        // even when a successor has a lower MIR block index than its predecessor.
        for &bb in &bfs_ordered_bbs {
            let bb_data = &mir.basic_blocks[bb];
            if loop_region_blocks.contains(&bb) {
                continue; // Handled inside codegen_scf_for_loop.
            }

            if let Some(&loop_idx) = loop_init_bb_map.get(&bb) {
                // Process init-block statements only, then build the scf.for in-place.
                let init_mlir_block = *basic_blocks.get(&bb).expect("init block");
                for stmt in &bb_data.statements {
                    self.codegen_statement(
                        tcx,
                        instance,
                        mir,
                        stmt,
                        &init_mlir_block,
                        &mut state,
                    )?;
                }
                self.codegen_scf_for_loop(
                    tcx,
                    instance,
                    mir,
                    &loop_infos[loop_idx],
                    &init_mlir_block,
                    &func_op,
                    &mut state,
                    &basic_blocks,
                    location,
                )?;
                continue;
            }

            self.codegen_basic_block(
                tcx,
                instance,
                &mir,
                bb,
                bb_data,
                &func_op,
                &mut state,
                &basic_blocks,
            )?;
        }

        println!("[DEBUG] TritonCodegen::codegen_function end: ssa_values: {:?}", state.ssa_values);
        self.module.mlir.body().append_operation(func_op.into());

        Ok(())
    }

    fn codegen_basic_block<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        _func_op: &Operation,
        state: &mut CodegenState<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
    ) -> Result<(), MlirError> {
        let mlir_block = basic_blocks.get(&bb).expect("block not found");

        // At join blocks, the phi block args represent merged values for locals.
        // Before overwriting, save the current ssa_values for each phi local so that
        // any predecessor block processed *after* this join (in DFS order) can still
        // find the correct "pre-join" value when building its branch to this block.
        // Then update ssa_values so subsequent statements in this and successor blocks
        // see the phi block arg value.
        if let Some(phi_locals) = state.phi_join_locals.get(&bb).cloned() {
            for phi_local in &phi_locals {
                if let Some(&phi_val) = state.phi_block_args.get(&(bb, *phi_local)) {
                    // Save old value before overwriting.
                    if let Some(&old_val) = state.ssa_values.get(phi_local) {
                        state.pre_join_ssa_values.entry((bb, *phi_local)).or_insert(old_val);
                    }
                    state.ssa_values.insert(*phi_local, phi_val);
                }
            }
        }

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, instance, mir, stmt, mlir_block, state)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(
            tcx,
            instance,
            mir,
            bb_data.terminator(),
            mlir_block,
            state,
            basic_blocks,
        )?;

        Ok(())
    }

    fn codegen_statement<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        //println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        let location = span_to_location(self.module.context(), tcx, stmt.source_info.span);
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                //println!("[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}", stmt, place, rvalue);
                self.codegen_assign(tcx, instance, mir, place, rvalue, location, mlir_block, state)
            }
            StatementKind::SetDiscriminant { place, variant_index } => self
                .codegen_set_discriminant(
                    tcx,
                    instance,
                    mir,
                    place,
                    *variant_index,
                    mlir_block,
                    state,
                ),
            StatementKind::StorageLive(local) => self.codegen_storage_live(tcx, *local, mlir_block),
            StatementKind::StorageDead(local) => self.codegen_storage_dead(tcx, *local, mlir_block),
            StatementKind::Intrinsic(intrinsic) => {
                self.codegen_intrinsic(tcx, intrinsic, mlir_block)
            }
            // Runtime no-ops or analysis-only statements that require no codegen.
            StatementKind::Nop
            | StatementKind::ConstEvalCounter
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::BackwardIncompatibleDropHint { .. }
            | StatementKind::Retag(..) => Ok(()),
        }?;

        //println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_assign<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        match rvalue {
            Rvalue::Use(operand) => {
                let ty = operand.ty(mir, tcx);
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(ty),
                );

                // Route Option<T> locals into the option_table instead of ssa_values.
                if is_option_ty(tcx, normalized_ty) {
                    let opt_value = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            *state.option_table.get(&src.local).unwrap_or_else(|| {
                                panic!("Option local {:?} not found in option_table", src.local)
                            })
                        }
                        // `None::<T>` as a constant — no inner value.
                        Operand::Constant(_) => None,
                        Operand::RuntimeChecks(_) => {
                            todo!("RuntimeChecks operand not yet supported")
                        }
                    };
                    state.option_table.insert(place.local, opt_value);
                    return Ok(());
                }

                // Route Tuple<T...> locals into tuple_fields.
                if let TyKind::Tuple(elem_tys) = normalized_ty.kind() {
                    let fields = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            state.tuple_fields.get(&src.local).cloned().unwrap_or_else(|| {
                                panic!("Tuple local {:?} not found in tuple_fields", src.local)
                            })
                        }
                        Operand::Constant(const_op) => self.codegen_tuple_constant(
                            tcx, instance, const_op, elem_tys, location, mlir_block,
                        )?,
                        Operand::RuntimeChecks(_) => todo!("RuntimeChecks for tuple"),
                    };
                    state.tuple_fields.insert(place.local, fields);
                    return Ok(());
                }

                // Propagate side-table entries for locals tracked outside ssa_values.
                // Copy/Move of an array/pointer/slice local just aliases the side-table entry.
                if let Operand::Copy(src) | Operand::Move(src) = operand {
                    if src.projection.is_empty() {
                        // dyn_array copy: _dest = copy _src where _src is in dyn_arrays.
                        if let Some(elems) = state.dyn_arrays.get(&src.local).cloned() {
                            state.dyn_arrays.insert(place.local, elems);
                            return Ok(());
                        }
                        // const_array copy.
                        if let Some(&arr_local) = state.ptr_to_const_array.get(&src.local) {
                            state.ptr_to_const_array.insert(place.local, arr_local);
                            return Ok(());
                        }
                        // dyn_array pointer copy.
                        if let Some(&arr_local) = state.ptr_to_dyn_array.get(&src.local) {
                            state.ptr_to_dyn_array.insert(place.local, arr_local);
                            return Ok(());
                        }
                        // slice_dyn_values copy.
                        if let Some(vals) = state.slice_dyn_values.get(&src.local).cloned() {
                            state.slice_dyn_values.insert(place.local, vals);
                            return Ok(());
                        }
                        // slice_shape copy.
                        if let Some(shape) = state.slice_shape.get(&src.local).cloned() {
                            state.slice_shape.insert(place.local, shape);
                            return Ok(());
                        }
                        // ADT/Range locals stored in tuple_fields (e.g. Range moves for for-loops).
                        if let Some(fields) = state.tuple_fields.get(&src.local).cloned() {
                            state.tuple_fields.insert(place.local, fields);
                            return Ok(());
                        }
                    }
                }

                // `_x = const &[T; N]` — populate slice_shape so PointerCoercion(Unsize)
                // can propagate the shape when this local is later coerced to &[T].
                if let Operand::Constant(c) = operand {
                    let const_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx, TypingEnv::fully_monomorphized(), EarlyBinder::bind(c.const_.ty()),
                    );
                    if let Some(shape) =
                        self.try_read_array_ref_const(tcx, *instance, const_ty, &c.const_)
                    {
                        state.slice_shape.insert(place.local, shape);
                        // Fall through to codegen_operand to emit a placeholder SSA value.
                    }
                }

                let result = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 1: result: Place: {:?}, Result: {:?}",
                    place, result
                );
                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                println!("Cast cast_kind: {:?}, operand: {:?}, ty: {:?}", cast_kind, operand, ty);

                // PtrToPtr cast on a const-array pointer — propagate the side-table entry
                // without emitting an MLIR op (const arrays have no MLIR value).
                if matches!(cast_kind, CastKind::PtrToPtr) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(&arr_local) = state.ptr_to_const_array.get(&src_place.local) {
                            state.ptr_to_const_array.insert(place.local, arr_local);
                            return Ok(());
                        }
                    }
                }

                // Transmute of a slice-shape tuple ((*const T, usize) → &[T]) — propagate shape.
                if matches!(cast_kind, CastKind::Transmute) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(shape) = state.slice_shape.get(&src_place.local).cloned() {
                            state.slice_shape.insert(place.local, shape);
                            return Ok(());
                        }
                    }
                }

                // PointerCoercion(Unsize) from a promoted &[T; N] constant → &[T].
                // e.g. `zeros::<D>(&[BLOCK_SIZE])` generates:
                //   _23 = const promoted[41] as &[i32] (PointerCoercion(Unsize, Implicit))
                // Extract the array values and populate slice_shape so downstream calls
                // (zeros, broadcast_to, reshape, etc.) can read the tensor shape.
                if let CastKind::PointerCoercion(PointerCoercion::Unsize, _) = cast_kind {
                    if let Operand::Constant(const_op) = operand {
                        let src_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                            tcx,
                            TypingEnv::fully_monomorphized(),
                            EarlyBinder::bind(const_op.const_.ty()),
                        );
                        if let Some(shape) =
                            self.try_read_array_ref_const(tcx, *instance, src_ty, &const_op.const_)
                        {
                            state.slice_shape.insert(place.local, shape);
                            // Fall through to codegen_cast to emit the i64 placeholder value.
                        }
                    } else if let Operand::Copy(p) | Operand::Move(p) = operand {
                        println!(
                            "[DEBUG-UNSIZE] PointerCoercion dynamic: place={:?} p.local={:?} proj_empty={} ptr_to_dyn={:?} ptr_to_const={:?}",
                            place.local, p.local, p.projection.is_empty(),
                            state.ptr_to_dyn_array.get(&p.local),
                            state.ptr_to_const_array.get(&p.local),
                        );
                        if p.projection.is_empty() {
                            // Dynamic pointer (from a dyn_array via Ref) → slice_dyn_values.
                            if let Some(&arr_local) = state.ptr_to_dyn_array.get(&p.local) {
                                if let Some(vals) = state.dyn_arrays.get(&arr_local).cloned() {
                                    println!("[DEBUG-UNSIZE] inserting slice_dyn_values[{:?}] = {} vals", place.local, vals.len());
                                    state.slice_dyn_values.insert(place.local, vals);
                                } else {
                                    println!("[DEBUG-UNSIZE] arr_local={:?} NOT in dyn_arrays", arr_local);
                                }
                            }
                            // Static pointer (from a const_array via Ref) → slice_shape.
                            if let Some(&arr_local) = state.ptr_to_const_array.get(&p.local) {
                                if let Some(shape) = state.const_arrays.get(&arr_local).cloned() {
                                    println!("[DEBUG-UNSIZE] inserting slice_shape[{:?}] = {:?}", place.local, shape);
                                    state.slice_shape.insert(place.local, shape);
                                }
                            }
                            // Const array ref assigned directly (e.g. `_x = const &[T; N]`) → slice_shape.
                            if let Some(shape) = state.slice_shape.get(&p.local).cloned() {
                                state.slice_shape.insert(place.local, shape);
                            }
                        }
                    }
                }

                let result = self.codegen_cast(
                    tcx, instance, cast_kind, operand, ty, location, mlir_block, state,
                )?;

                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 2: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign: Aggregate: {:?}, index_vec: {:?}",
                    aggregate_kind, index_vec
                );
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign: ssa_values: {:?}",
                    state.ssa_values
                );

                // Route Option<T> aggregates (Some/None) into the option_table.
                if let AggregateKind::Adt(def_id, variant_index, _, _, _) = aggregate_kind.as_ref()
                {
                    let adt_def = tcx.adt_def(*def_id);
                    let norm_place_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(place.ty(mir, tcx).ty),
                    );
                    if is_option_ty(tcx, norm_place_ty) {
                        return self.codegen_option_aggregate(
                            tcx,
                            instance,
                            mir,
                            place,
                            adt_def,
                            *variant_index,
                            index_vec,
                            location,
                            mlir_block,
                            state,
                        );
                    }
                }

                // Route Tuple aggregates into tuple_fields (MLIR uses multiple return values).
                if let AggregateKind::Tuple = aggregate_kind.as_ref() {
                    // Special case: `(*const T, usize)` fat-pointer tuple where the pointer
                    // comes from a const array.  Route to `slice_shape` instead — no MLIR value.
                    if let Some(first_op) = index_vec.iter().next() {
                        if let Operand::Copy(p) | Operand::Move(p) = first_op {
                            if let Some(&arr_local) = state.ptr_to_const_array.get(&p.local) {
                                if let Some(shape) = state.const_arrays.get(&arr_local) {
                                    state.slice_shape.insert(place.local, shape.clone());
                                    return Ok(());
                                }
                            }
                            if let Some(&arr_local) = state.ptr_to_dyn_array.get(&p.local) {
                                if let Some(vals) = state.dyn_arrays.get(&arr_local).cloned() {
                                    state.slice_dyn_values.insert(place.local, vals);
                                    return Ok(());
                                }
                            }
                        }
                    }

                    let fields = index_vec
                        .iter()
                        .map(|op| match op {
                            Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(&p, state),
                            _ => todo!("Tuple aggregate with non-copy/move operand: {:?}", op),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    state.tuple_fields.insert(place.local, fields);
                    return Ok(());
                }

                // Constant integer arrays (e.g. shape arrays `[BLOCK_SIZE]` for zeros/reshape).
                // We only track them as constant metadata; no MLIR value is emitted.
                if let AggregateKind::Array(_elem_ty) = aggregate_kind.as_ref() {
                    let elems: Option<Vec<i64>> = index_vec
                        .iter()
                        .map(|op| {
                            self.to_scalar_int(tcx, instance, op).ok().map(|s| {
                                match s.size().bytes() {
                                    1 => s.to_u8() as i64,
                                    2 => s.to_i16() as i64,
                                    4 => s.to_i32() as i64,
                                    8 => s.to_i64(),
                                    n => todo!("ScalarInt size {} bytes", n),
                                }
                            })
                        })
                        .collect();
                    if let Some(elems) = elems {
                        state.const_arrays.insert(place.local, elems);
                        return Ok(());
                    }
                    // Non-constant elements: look up each as an SSA value.
                    let dyn_elems: Result<Vec<Value<'_, '_>>, MlirError> = index_vec
                        .iter()
                        .map(|op| match op {
                            Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(p, state),
                            Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                                    tcx,
                                    TypingEnv::fully_monomorphized(),
                                    EarlyBinder::bind(op.ty(mir, tcx)),
                                );
                                self.codegen_operand(
                                    tcx, instance, op, ty, location, mlir_block, state,
                                )
                            }
                        })
                        .collect();
                    let dyn_elems = dyn_elems?;
                    println!("[DEBUG-ARR] dyn_arrays[{:?}] = {} elems", place.local, dyn_elems.len());
                    state.dyn_arrays.insert(place.local, dyn_elems);
                    return Ok(());
                }

                // Generic ADT structs (e.g. core::ops::Range, custom structs): treat fields as
                // a "tuple" stored in tuple_fields so that field-projection reads work.
                // Known Triton types are handled inside codegen_aggregate_create.
                if let AggregateKind::Adt(def_id, _, _, _, _) = aggregate_kind.as_ref() {
                    let adt_def = tcx.adt_def(*def_id);
                    let adt_name = format!("{:?}", adt_def);
                    let is_triton_type = adt_name == "triton::llvm::triton::tensor::LlvmTensor"
                        || adt_name == "triton::llvm::triton::pointer::LlvmPointer";
                    if !is_triton_type {
                        let fields: Result<Vec<Value<'_, '_>>, MlirError> = index_vec
                            .iter()
                            .map(|op| match op {
                                Operand::Copy(p) | Operand::Move(p) => {
                                    self.codegen_copy(p, state)
                                }
                                Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                                    let ty =
                                        instance.instantiate_mir_and_normalize_erasing_regions(
                                            tcx,
                                            TypingEnv::fully_monomorphized(),
                                            EarlyBinder::bind(op.ty(mir, tcx)),
                                        );
                                    self.codegen_operand(
                                        tcx, instance, op, ty, location, mlir_block, state,
                                    )
                                }
                            })
                            .collect();
                        state.tuple_fields.insert(place.local, fields?);
                        return Ok(());
                    }
                }

                let result = self.codegen_aggregate_create(
                    tcx,
                    instance,
                    mir,
                    aggregate_kind,
                    index_vec,
                    location,
                    mlir_block,
                    state,
                )?;
                println!(
                    "[DEBUG] TritonCodegen::codegen_assign ssa_values_insert 3: result: Place: {:?}, Result: {:?}",
                    place, result
                );

                if let Some(result) = result {
                    println!(
                        "codegen_aggregate_create: result: ** {:?} ** {:?}",
                        place.local, result
                    );
                    state.ssa_values.insert(place.local, result);
                } else {
                    println!(
                        "[DEBUG] TritonCodegen::codegen_assign: result is None: {:?} {:?}",
                        place.local, rvalue
                    );
                }
            }
            Rvalue::Repeat(operand, _) => todo!("Repeat: {:?}", operand),
            Rvalue::Ref(_region, _borrow_kind, src_place) => {
                // Treat `&arr` the same as `&raw const arr` for const/dyn arrays so that
                // downstream slice-coercion and `make_tensor_descriptor` can recover the values.
                if src_place.projection.is_empty() {
                    if state.const_arrays.contains_key(&src_place.local) {
                        println!("[DEBUG-REF] ptr_to_const_array[{:?}] = {:?}", place.local, src_place.local);
                        state.ptr_to_const_array.insert(place.local, src_place.local);
                        return Ok(());
                    }
                    if state.dyn_arrays.contains_key(&src_place.local) {
                        println!("[DEBUG-REF] ptr_to_dyn_array[{:?}] = {:?}", place.local, src_place.local);
                        state.ptr_to_dyn_array.insert(place.local, src_place.local);
                        return Ok(());
                    }
                }
                todo!("Ref: {:?} {:?} {:?}", _region, _borrow_kind, src_place)
            }
            Rvalue::ThreadLocalRef(def_id) => todo!("ThreadLocalRef: {:?}", def_id),
            Rvalue::RawPtr(_raw_ptr_kind, src_place) => {
                // If this is `&raw const const_array_local`, record the mapping so that
                // downstream `zeros` / `slice_from_raw_parts` calls can recover the shape.
                if src_place.projection.is_empty() {
                    if state.const_arrays.contains_key(&src_place.local) {
                        state.ptr_to_const_array.insert(place.local, src_place.local);
                        return Ok(());
                    }
                    if state.dyn_arrays.contains_key(&src_place.local) {
                        state.ptr_to_dyn_array.insert(place.local, src_place.local);
                        return Ok(());
                    }
                }
                todo!("RawPtr: {:?} {:?}", _raw_ptr_kind, src_place)
            }
            Rvalue::BinaryOp(bin_op, operands) => {
                // WithOverflow variants return (T, bool) — treat as plain arithmetic + false flag.
                let is_overflow_op = matches!(
                    bin_op,
                    BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
                );
                if is_overflow_op {
                    let plain_op = match bin_op {
                        BinOp::AddWithOverflow => BinOp::Add,
                        BinOp::SubWithOverflow => BinOp::Sub,
                        BinOp::MulWithOverflow => BinOp::Mul,
                        _ => unreachable!(),
                    };
                    let value = self.codegen_binary_op(
                        tcx, instance, mir, place, &plain_op, operands, location, mlir_block, state,
                    )?;
                    if let Some(result_val) = value {
                        let ctx = self.module.context();
                        let i1_ty = IntegerType::new(ctx, 1);
                        let false_attr = IntegerAttribute::new(i1_ty.into(), 0);
                        let false_op: Operation<'a> =
                            melior::dialect::arith::constant(ctx, false_attr.into(), location)
                                .into();
                        let false_val: Value = false_op.result(0).unwrap().into();
                        mlir_block.append_operation(false_op);
                        state.tuple_fields.insert(place.local, vec![result_val, false_val]);
                    }
                } else {
                    let value = self.codegen_binary_op(
                        tcx, instance, mir, place, bin_op, operands, location, mlir_block, state,
                    )?;
                    if let Some(value) = value {
                        state.ssa_values.insert(place.local, value);
                    }
                }
            }
            Rvalue::UnaryOp(un_op, operand) => todo!("UnaryOp: {:?} {:?}", un_op, operand),
            Rvalue::Discriminant(src_place) => {
                let src_ty = src_place.ty(mir, tcx).ty;
                let norm_src_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(src_ty),
                );
                if is_option_ty(tcx, norm_src_ty) {
                    // The discriminant of an Option is statically known from the option_table.
                    let discr: i64 = match state.option_table.get(&src_place.local) {
                        Some(None) => 0,    // None variant
                        Some(Some(_)) => 1, // Some variant
                        None => panic!(
                            "Option local {:?} not found in option_table for Discriminant",
                            src_place.local
                        ),
                    };
                    let int_val = Int::I8(discr as u8);
                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, int_val)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = const_op
                        .result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    state.ssa_values.insert(place.local, result.into());
                    return Ok(());
                }
                todo!("Discriminant for non-Option: {:?}", src_place)
            }
            Rvalue::ShallowInitBox(operand, ty) => todo!("ShallowInitBox: {:?} {:?}", operand, ty),
            Rvalue::CopyForDeref(place) => todo!("CopyForDeref: {:?}", place),
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                todo!("WrapUnsafeBinder: {:?} {:?}", operand, ty)
            }
        }

        // todo!("[TODO] TritonCodegen::codegen_assign: {:?} {:?}", place, rvalue)
        Ok(())
    }

    fn codegen_aggregate_create<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!("codegen_aggregate_assign: {:?} {:?} {:?}", aggregate_kind, index_vec, mlir_block);

        match aggregate_kind {
            AggregateKind::Adt(def_id, _, raw_list, _, _) => {
                // Get the ADT definition and a human-readable name for debugging.
                let adt_def = tcx.adt_def(*def_id);
                let adt_name = format!("{:?}", adt_def);

                if "triton::llvm::triton::tensor::LlvmTensor" == adt_name {
                    self.codegen_create_tensor(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else if "triton::llvm::triton::pointer::LlvmPointer" == adt_name {
                    self.codegen_create_pointer(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else {
                    todo!(
                        "codegen_aggregate_create: {:?} {:?} {:?}",
                        adt_name,
                        adt_def,
                        index_vec.as_slice()
                    );
                }
            }
            _ => todo!("AggregateKind: {:?}", aggregate_kind),
        }
    }

    fn codegen_create_tensor<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(arg1.ty(mir, tcx)),
        );

        let pointee_ty = match arg1_ty.kind() {
            TyKind::RawPtr(pointee_ty, _) => {
                self.type_mapper.map_type(self.module.context(), &tcx, &pointee_ty)
            }
            _ => todo!("codegen_create_tensor: arg1_ty: {:?}", arg1_ty),
        };
        let tensor_type = tensor_type(&[i64::MIN], pointee_ty).into();
        let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
            .map_err(|e| MlirError::CreateOperation { err: e })?;

        let tensor_result = tensor_op.result(0).unwrap();
        mlir_block.append_operation(tensor_op);
        Ok(Some(tensor_result.into()))
    }

    fn codegen_create_pointer<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(arg1.ty(mir, tcx)),
        );
        // `Pointer<T>` is a newtype wrapper around a raw pointer in Triton DSL.
        // Preserve the wrapped pointer SSA value rather than materializing poison.
        let pointer_value =
            self.codegen_operand(tcx, instance, arg1, arg1_ty, location, mlir_block, state)?;
        Ok(Some(pointer_value))
    }

    fn codegen_const_adt<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        adt_def: &AdtDef<'tcx>,
        value: Value<'a, 'a>,
        raw_list: &[GenericArg<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let name = format!("{:?}", adt_def);
        let map_ty = |idx: usize| {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(raw_list[idx].expect_ty()),
            );
            self.type_mapper.map_type(self.module.context(), &tcx, &ty)
        };

        // If the name of the ADT is tensor, then we create a poison operation.
        // This is because the tensor creation is part of the dsl dead code which
        // will be eliminated by the optimizer.
        if name == "triton::llvm::triton::tensor::Tensor" {
            let ty = map_ty(0);
            let tensor_type = tensor_type(&[i64::MIN], ty).into();
            let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
                .map_err(|e| MlirError::CreateOperation { err: e })?;

            let tensor_result = tensor_op.result(0).unwrap();
            mlir_block.append_operation(tensor_op);
            Ok(tensor_result.into())
        } else if name == "triton::llvm::triton::pointer::Pointer" {
            // `Pointer<T>` is a newtype wrapper — pass through the wrapped value directly.
            Ok(value)
        } else if adt_def.is_enum() {
            // For enum ADTs (e.g. `Axis`, `PaddingOption`, `CacheModifier`, etc.),
            // `codegen_scalar_const_value` already emitted the discriminant as an integer.
            // Pass through the value unchanged.
            Ok(value)
        } else {
            todo!("Adt: {:?}", adt_def)
        }
    }

    fn codegen_binary_op<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        bin_op: &BinOp,
        operands: &(Operand<'tcx>, Operand<'tcx>),
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let (lhs_op, rhs_op) = operands;

        let lhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(lhs_op.ty(mir, tcx)),
        );
        let rhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(rhs_op.ty(mir, tcx)),
        );
        let lhs =
            self.codegen_operand(tcx, instance, lhs_op, lhs_ty, location, mlir_block, state)?;
        let rhs =
            self.codegen_operand(tcx, instance, rhs_op, rhs_ty, location, mlir_block, state)?;

        match bin_op {
            BinOp::Add | BinOp::AddUnchecked => {
                self.codegen_add(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::AddWithOverflow => todo!("BinOp::AddWithOverflow"),
            BinOp::Sub | BinOp::SubUnchecked => {
                self.codegen_sub(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::SubWithOverflow => todo!("BinOp::SubWithOverflow"),
            BinOp::Mul | BinOp::MulUnchecked => {
                self.codegen_mul(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::MulWithOverflow => todo!("BinOp::MulWithOverflow"),
            BinOp::Div => self.codegen_div(tcx, location, lhs, rhs, mlir_block),
            BinOp::Rem => self.codegen_rem(tcx, location, lhs, rhs, mlir_block),
            BinOp::BitXor => self.codegen_xor(tcx, location, lhs, rhs, mlir_block),
            BinOp::BitAnd => self.codegen_and(tcx, location, lhs, rhs, mlir_block),
            BinOp::BitOr => self.codegen_or(tcx, location, lhs, rhs, mlir_block),
            BinOp::Shl | BinOp::ShlUnchecked => {
                self.codegen_shl(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::Shr | BinOp::ShrUnchecked => {
                let signed = matches!(lhs_ty.kind(), TyKind::Int(_));
                self.codegen_shr(tcx, signed, location, lhs, rhs, mlir_block)
            }
            BinOp::Eq => self.codegen_cmpi(tcx, Predicate::EQ, location, lhs, rhs, mlir_block),
            BinOp::Lt => self.codegen_cmpi(tcx, Predicate::SLT, location, lhs, rhs, mlir_block),
            BinOp::Le => self.codegen_cmpi(tcx, Predicate::SLE, location, lhs, rhs, mlir_block),
            BinOp::Ne => self.codegen_cmpi(tcx, Predicate::NE, location, lhs, rhs, mlir_block),
            BinOp::Ge => self.codegen_cmpi(tcx, Predicate::SGE, location, lhs, rhs, mlir_block),
            BinOp::Gt => self.codegen_cmpi(tcx, Predicate::SGT, location, lhs, rhs, mlir_block),
            BinOp::Cmp => todo!("BinOp::Cmp"),
            BinOp::Offset => todo!("BinOp::Offset"),
        }
    }

    /// Codegen construction of an `Option<T>` aggregate (either `None` or `Some(inner)`).
    /// The result is stored in `state.option_table` rather than `state.ssa_values`.
    #[allow(clippy::too_many_arguments)]
    fn codegen_option_aggregate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        _adt_def: rustc_middle::ty::AdtDef<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        index_vec: &rustc_index::IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        if variant_index.as_usize() == 0 {
            // None variant — no inner value.
            state.option_table.insert(place.local, None);
        } else {
            // Some(inner) — codegen the single field and stash it.
            let inner_op = index_vec
                .get(FieldIdx::from_usize(0))
                .expect("Option::Some aggregate must have exactly one field");
            let inner_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(inner_op.ty(mir, tcx)),
            );
            let inner_value = self
                .codegen_operand(tcx, instance, inner_op, inner_ty, location, mlir_block, state)?;
            state.option_table.insert(place.local, Some(inner_value));
        }
        Ok(())
    }

    fn codegen_set_discriminant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        _mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let place_ty = place.ty(mir, tcx).ty;
        let norm_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(place_ty),
        );

        if is_option_ty(tcx, norm_ty) {
            if variant_index.as_usize() == 0 {
                // Two-phase None: discriminant set to 0.
                state.option_table.insert(place.local, None);
            } else {
                // Two-phase Some: inner value was written into ssa_values[place.local] by
                // a preceding field-assignment via Place::Downcast projection.  Promote it.
                let inner = state.ssa_values.remove(&place.local).unwrap_or_else(|| {
                    panic!(
                        "SetDiscriminant Some: expected inner value in ssa_values for {:?}",
                        place.local
                    )
                });
                state.option_table.insert(place.local, Some(inner));
            }
            return Ok(());
        }

        todo!("[TODO] TritonCodegen::codegen_set_discriminant for non-Option types")
    }

    fn codegen_storage_live<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_storage_live: local: {:?}", _local);
        // NO-OP: In the context of Triton and MLIR, storage live is a no-op.
        Ok(())
    }

    fn codegen_storage_dead<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        println!("[DEBUG] TritonCodegen::codegen_storage_dead: local: {:?}", _local);
        // NO-OP: In the context of Triton and MLIR, storage dead is a no-op.
        Ok(())
    }

    fn codegen_intrinsic<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _intrinsic: &NonDivergingIntrinsic<'tcx>,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_intrinsic")
    }

    fn codegen_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        cast_kind: &CastKind,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match cast_kind {
            CastKind::PointerWithExposedProvenance => self.codegen_pointer_with_exposed_provenance(
                tcx, instance, operand, ty, location, mlir_block, state,
            ),
            CastKind::PtrToPtr => {
                self.codegen_ptr_to_ptr(tcx, instance, operand, ty, location, mlir_block, state)
            }
            CastKind::IntToInt => {
                self.codegen_int_to_int(tcx, instance, operand, ty, location, mlir_block, state)
            }
            _ => {
                // Unhandled cast kinds (Transmute, PointerCoercion, ReifyFnPointer, etc.).
                // Emit ub.poison of the destination type as a safe placeholder.
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(*ty),
                );
                println!(
                    "[DEBUG] codegen_cast fallback: cast_kind={:?} ty={:?}",
                    cast_kind, normalized_ty
                );
                let mlir_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                let ub_op: Operation<'a> =
                    create_ub_poison(self.module.context(), location, mlir_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = ub_op.result(0).expect("ub.poison result").into();
                mlir_block.append_operation(ub_op);
                Ok(result)
            }
        }
    }

    fn codegen_pointer_with_exposed_provenance<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        println!(
            "[DEBUG] TritonCodegen::codegen_pointer_with_exposed_provenance: provenance: {:?} ty: {:?} normalized: {:?}",
            operand, ty, normalized_ty
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_ptr_to_ptr<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_int_to_int<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(*ty),
        );
        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        //println!("[DEBUG] TritonCodegen::codegen_operand: ssa_values: {:?} operand: {:?}", state.ssa_values, operand);

        // For MLIR move is the same as copy
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.codegen_copy(place, state),
            Operand::Constant(const_operand) => self.codegen_constant_cast(
                tcx,
                instance,
                const_operand,
                normalized_ty,
                location,
                mlir_block,
            ),
            Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
        }
    }

    /// Resolve an operand that may be `Option<T>`.
    /// Returns the inner MLIR value if the option is `Some` (or if the operand is not an Option),
    /// or `None` if the option is absent.
    pub(crate) fn codegen_option_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(operand.ty(mir, tcx)),
        );

        if is_option_ty(tcx, ty) {
            let inner = match operand {
                Operand::Copy(p) | Operand::Move(p) => {
                    *state.option_table.get(&p.local).unwrap_or_else(|| {
                        panic!("Option local {:?} not found in option_table", p.local)
                    })
                }
                // `None::<T>` constant — no inner value.
                Operand::Constant(_) => None,
                Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
            };
            Ok(inner)
        } else {
            let value =
                self.codegen_operand(tcx, instance, operand, ty, location, mlir_block, state)?;
            Ok(Some(value))
        }
    }

    /// Decode a constant tuple into individual MLIR scalar values.
    /// Reads each element from the underlying memory allocation using layout offsets.
    fn codegen_tuple_constant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_op: &ConstOperand<'tcx>,
        elem_tys: &[Ty<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let normalized_const = match &const_op.const_ {
            Const::Val(const_val, _) => *const_val,
            other => todo!("codegen_tuple_constant: unexpected const kind {:?}", other),
        };

        let (alloc_id, base_offset) = match normalized_const {
            ConstValue::Indirect { alloc_id, offset } => (alloc_id, offset),
            other => todo!("codegen_tuple_constant: unexpected ConstValue {:?}", other),
        };

        let GlobalAlloc::Memory(const_alloc) = tcx.global_alloc(alloc_id) else {
            todo!("codegen_tuple_constant: non-memory alloc for {:?}", alloc_id);
        };
        let alloc = const_alloc.inner();

        // Get the layout of the full tuple to obtain per-field offsets.
        let tuple_ty = const_op.const_.ty();
        let tuple_layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(tuple_ty))
            .map_err(|e| MlirError::CodegenFailed { err: format!("layout_of failed: {:?}", e) })?;

        let FieldsShape::Arbitrary { ref offsets, .. } = tuple_layout.fields else {
            todo!("codegen_tuple_constant: unexpected FieldsShape for tuple");
        };

        let mut values = Vec::with_capacity(elem_tys.len());
        for (field_idx, field_ty) in elem_tys.iter().enumerate() {
            let normalized_field_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(*field_ty),
            );
            let field_layout = tcx
                .layout_of(TypingEnv::fully_monomorphized().as_query_input(normalized_field_ty))
                .map_err(|e| MlirError::CodegenFailed {
                    err: format!("layout_of field failed: {:?}", e),
                })?;

            let field_offset = base_offset + offsets[FieldIdx::from_usize(field_idx)];
            let range = alloc_range(field_offset, field_layout.size);
            let scalar = alloc
                .read_scalar(&tcx, range, false)
                .map_err(|e| MlirError::CodegenFailed { err: format!("read_scalar: {:?}", e) })?;

            let value = self.codegen_scalar(normalized_field_ty, scalar, location, mlir_block)?;
            values.push(value);
        }

        Ok(values)
    }

    fn codegen_copy<'tcx>(
        &self,
        place: &Place<'tcx>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        eprintln!("[DEBUG-COPY] codegen_copy: Local: {:?}, proj_len={}, in_ssa={}, in_dyn={}, in_ptr_dyn={}, in_slice_dyn={}",
            place.local, place.projection.len(),
            state.ssa_values.contains_key(&place.local),
            state.dyn_arrays.contains_key(&place.local),
            state.ptr_to_dyn_array.contains_key(&place.local),
            state.slice_dyn_values.contains_key(&place.local));

        // Handle Option downcast+field projection: `(_opt as Some).0` — extract the inner value.
        if let [ProjectionElem::Downcast(_, _), ProjectionElem::Field(_, _)]
        | [ProjectionElem::Field(_, _)] = place.projection.as_slice()
        {
            if let Some(opt_val) = state.option_table.get(&place.local) {
                return Ok(opt_val.unwrap_or_else(|| {
                    panic!(
                        "Accessing field of None Option local {:?}",
                        place.local
                    )
                }));
            }
        }

        debug_assert!(
            !state.option_table.contains_key(&place.local),
            "BUG: Option local {:?} used as a direct MLIR value; use codegen_option_operand instead",
            place.local
        );

        // Handle a single Field projection on a tuple local stored in tuple_fields.
        if let [ProjectionElem::Field(field_idx, _)] = place.projection.as_slice() {
            if let Some(fields) = state.tuple_fields.get(&place.local) {
                let idx = field_idx.index();
                return Ok(*fields.get(idx).unwrap_or_else(|| {
                    panic!("Tuple field {} not found for local {:?}", idx, place.local)
                }));
            }
        }

        Ok(state
            .ssa_values
            .get(&place.local)
            .copied()
            .expect(format!("Value not found for local: {:?}", place.local).as_str()))
    }

    fn codegen_constant_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_operand: &ConstOperand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_constant_cast: {:?}, {:?}",
            const_operand, normalized_ty
        );

        match const_operand.const_ {
            Const::Val(const_val, ty) => {
                let value =
                    self.codegen_const_value(tcx, instance, const_val, ty, location, mlir_block)?;

                match normalized_ty.kind() {
                    TyKind::RawPtr(_, _) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: RawPtr");
                        let value_ty = value.r#type();
                        debug_assert!(
                            value_ty.is_integer(),
                            "Triton supports only integer pointer casts"
                        );
                        let ptr_ty =
                            self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                        let cast_op: Operation =
                            int_to_ptr(self.module.context(), location, value.into(), ptr_ty)
                                .map_err(|e| MlirError::CreateOperation { err: e })?
                                .into();

                        let result = cast_op.result(0).unwrap();
                        mlir_block.append_operation(cast_op);
                        Ok(result.into())
                    }
                    TyKind::Adt(adt_def, args) => {
                        println!("[DEBUG] TritonCodegen::codegen_constant_cast: Adt");
                        let result = self.codegen_const_adt(
                            tcx,
                            instance,
                            adt_def,
                            value,
                            args.as_slice(),
                            location,
                            mlir_block,
                        )?;
                        Ok(result)
                    }
                    TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::Bool => {
                        // Constant already has the right primitive type — return as-is.
                        Ok(value)
                    }
                    TyKind::Ref(_, _, _) | TyKind::Str => {
                        // &str, &[T], and other reference types are represented as i64
                        // fat-pointer stand-ins in the type mapper. The value already
                        // carries the right i64 type from codegen_const_value.
                        Ok(value)
                    }
                    TyKind::FnPtr(_, _) | TyKind::FnDef(_, _) => {
                        // Function pointer types are represented as i64 — return as-is.
                        Ok(value)
                    }
                    _ => todo!("Constant cast normalized_ty: {:?}", normalized_ty),
                }
            }
            Const::Ty(ty, const_val) => match const_val.kind() {
                ConstKind::Param(param) => {
                    self.codegen_param_const(tcx, instance, ty, param, location, mlir_block)
                }
                ConstKind::Infer(_infer_const) => todo!("ConstKind::Infer"),
                ConstKind::Bound(_bound_var_index_kind, _) => todo!("ConstKind::Bound"),
                ConstKind::Placeholder(_) => todo!("ConstKind::Placeholder"),
                ConstKind::Unevaluated(_unevaluated_const) => todo!("ConstKind::Unevaluated"),
                ConstKind::Value(_) => todo!("ConstKind::Value"),
                ConstKind::Error(_) => todo!("ConstKind::Error"),
                ConstKind::Expr(_) => todo!("ConstKind::Expr"),
            },
            _ => todo!("Const: {:?}", const_operand.const_),
        }
    }

    fn codegen_param_const<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _ty: Ty<'tcx>,
        param: ParamConst,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let value = instance.args.const_at(param.index as usize).to_value();
        self.codegen_param_value(tcx, value, location, mlir_block)
    }

    fn codegen_param_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        value: ty::Value<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let scalar = value.try_to_scalar();

        if let Some(scalar) = scalar {
            self.codegen_scalar(value.ty, scalar, location, mlir_block)
        } else {
            todo!("codegen_param_value: {:?} scalar: {:?}", value, scalar);
        }
    }

    fn codegen_scalar<'tcx>(
        &self,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => {
                let scalar_attr = create_scalar_attr(self.module.context(), ty, scalar_int)
                    .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op =
                    create_constant(self.module.context(), location, scalar_attr.0, scalar_attr.1)
                        .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op: Operation = op.into();
                let result = op.result(0).expect("Constant operation result not found");
                mlir_block.append_operation(op);
                Ok(result.into())
            }
            Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
        }
    }

    fn codegen_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_val: ConstValue,
        ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match const_val {
            ConstValue::Scalar(scalar) => {
                self.codegen_scalar_const_value(tcx, instance, ty, scalar, location, mlir_block)
            }
            ConstValue::ZeroSized => {
                // Zero-sized values (fn items, unit, zero-sized structs).
                // Emit i64(0) as a harmless placeholder — these are never read
                // by any real MLIR op in the stub-based codegen.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("zero const").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Slice { alloc_id: _, meta: _ } => {
                // String literals and other slice constants (&str, &[T]).
                // The type mapper maps &str and &[T] to i64 (fat-pointer stand-in).
                // Emit i64(0) as a placeholder; the callee (device_print etc.) is a stub.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("slice const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Indirect { alloc_id: _, offset: _ } => {
                // Indirect constants (references to static allocations).
                // Emit i64(0) as a placeholder; these are not exercised by GPU stubs.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("indirect const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
        }
    }

    fn codegen_scalar_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => match ty.kind() {
                TyKind::Uint(_) | TyKind::Int(_) => {
                    let value =
                        Int::from_scalar(ty, scalar_int).map_err(|e| MlirError::InvalidScalar {
                            node: format!("Invalid scalar: {:?} {:?} {:?}", e, ty, scalar_int),
                        })?;

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, value)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();

                    let result = const_op
                        .result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Adt(adt_def, args) => {
                    println!(
                        "[DEBUG] TritonCodegen::codegen_scalar_const_value: Adt: {:?} {:?} {:?}",
                        scalar, adt_def, args
                    );

                    let scalar_int = match scalar {
                        Scalar::Int(s) => s,
                        Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
                    };

                    // For enum ADTs (e.g. `#[repr(i32)] enum Axis { X=0, Y=1, Z=2 }`),
                    // emit the discriminant as the underlying integer directly.
                    let int_val = if adt_def.is_enum() {
                        match scalar_int.size().bytes() {
                            1 => Int::I8(scalar_int.to_u8()),
                            2 => Int::I16(scalar_int.to_u16()),
                            4 => Int::I32(scalar_int.to_u32()),
                            8 => Int::I64(scalar_int.to_u64()),
                            n => todo!("Enum scalar size {} bytes", n),
                        }
                    } else {
                        // For scalar newtype ADTs (e.g. `struct I32(pub i32)`), get the inner
                        // field's primitive type so Int::from_scalar can determine the
                        // correct MLIR integer kind.
                        let variant = adt_def.non_enum_variant();
                        let inner_ty = tcx
                            .type_of(variant.fields[FieldIdx::from_usize(0)].did)
                            .instantiate(tcx, args);
                        Int::from_scalar(inner_ty, scalar_int).map_err(|e| {
                            MlirError::InvalidScalar {
                                node: format!(
                                    "Invalid scalar: {:?} {:?} {:?}",
                                    e, inner_ty, scalar_int
                                ),
                            }
                        })?
                    };

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, int_val)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Bool => {
                    // bool → i1 constant (0 = false, 1 = true)
                    let val = scalar_int.to_u8();
                    let i1_ty = IntegerType::new(self.module.context(), 1);
                    let attr = IntegerAttribute::new(i1_ty.into(), val as i64);
                    let const_op: Operation<'a> = melior::dialect::arith::constant(
                        self.module.context(),
                        attr.into(),
                        location,
                    )
                    .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                _ => todo!("Scalar::Int ty: {:?} {:?}", ty.kind(), ty),
            },
            Scalar::Ptr(ptr, size) => todo!("Ptr ptr: {:?}, size: {:?}", ptr, size),
        }
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                let fn_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
                let is_fn_ty = matches!(
                    fn_ty.kind(),
                    rustc_middle::ty::TyKind::FnDef(..) | rustc_middle::ty::TyKind::FnPtr(_, _)
                );

                if !is_fn_ty {
                    todo!(
                        "[DEBUG] TritonCodegen: instance.ty(tcx) is not a function type: {:?}",
                        fn_ty
                    );
                }

                self.codegen_function(tcx, fn_ty, instance)
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
