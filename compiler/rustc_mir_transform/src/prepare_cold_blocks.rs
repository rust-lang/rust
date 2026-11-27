//! MIR pass (`PrepareColdBlocks`) that moves shared borrows of hot `Copy` locals off the hot path
//! when the borrow only happens on the way to a panic.
//!
//! `panic!("{x}")`, `assert!(c, "{x}")` and similar macros take `&x` only on the panic path, but
//! `rustc_codegen_ssa::mir::analyze::non_ssa_locals` sees that borrow and gives `x` a stack slot
//! (`LocalRef::Place`) for the whole function, so the hot path pays for spilling `x`. This pass
//! copies `x` into a fresh local right before the borrow and borrows the copy instead, which lets
//! codegen keep `x` in a register and confines the stack traffic to the panic path.
//!
//! The rewrite must not be observable, so a local is only rewritten when:
//! - every borrow of it in the body is a shared borrow of the bare local: there is no `&raw`,
//!   `&mut` or field borrow through which a write could miss the original, and no other pointer to
//!   the local whose address could be compared with the copy's;
//! - each of those borrows is in a cold block that can never reach a `return`, is not on a cycle,
//!   and cannot reach another borrow of the same local, so at most one copy is taken per call;
//! - its type is `Copy`, `Freeze` and one or two scalars without padding, so the copy has the same
//!   bytes as the original for as long as the borrow lives. These are also exactly the types that
//!   codegen could keep in registers.

use std::cmp::Reverse;

use rustc_abi::BackendRepr;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::bug;
use tracing::debug;

use crate::PassPolicy;

pub(super) struct PrepareColdBlocks;

impl<'tcx> crate::MirPass<'tcx> for PrepareColdBlocks {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        // The point is faster optimized code; in unoptimized builds the copies only add work.
        PassPolicy::optional(ctx.mir_opt_level() >= 2)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        if tcx.is_coroutine(def_id) {
            return;
        }

        let cold_blocks = find_cold_blocks(tcx, body);
        if !cold_blocks.iter().any(|&is_cold| is_cold) {
            return;
        }

        if decouple_cold_borrows(tcx, body, &cold_blocks) {
            debug!(?def_id, "PrepareColdBlocks decoupled cold borrows in MIR body");
        }
    }
}

/// Identifies cold basic blocks in `mir`, including blocks ending in `Unreachable` and calls to
/// `#[cold]` functions, then propagates coldness backwards through postorder traversal across
/// non-cleanup edges.
pub(crate) fn find_cold_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir: &Body<'tcx>,
) -> IndexVec<BasicBlock, bool> {
    let local_decls = &mir.local_decls;
    let mut cold_blocks: IndexVec<BasicBlock, bool> = IndexVec::from_elem(false, &mir.basic_blocks);

    for (bb, bb_data) in traversal::postorder(mir) {
        if bb_data.is_cleanup {
            continue;
        }
        let terminator = bb_data.terminator();

        match terminator.kind {
            TerminatorKind::Call { ref func, .. } | TerminatorKind::TailCall { ref func, .. }
                if let ty::FnDef(def_id, ..) = *func.ty(local_decls, tcx).kind()
                    && tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::COLD) =>
            {
                cold_blocks[bb] = true;
                continue;
            }

            TerminatorKind::Unreachable => {
                cold_blocks[bb] = true;
                continue;
            }

            _ => {}
        }

        // Propagate coldness backwards across normal (non-cleanup) edges.
        let mut normal_succs = terminator.successors().filter(|&s| !mir.basic_blocks[s].is_cleanup);
        if let Some(first) = normal_succs.next()
            && cold_blocks[first]
            && normal_succs.all(|s| cold_blocks[s])
        {
            cold_blocks[bb] = true;
        }
    }

    cold_blocks
}

/// Copies hot locals into fresh locals right before they are borrowed on a path that can only end
/// in a panic (or otherwise never returns), and borrows the copies instead. See the module docs
/// for the conditions that keep this unobservable.
fn decouple_cold_borrows<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    cold_blocks: &IndexVec<BasicBlock, bool>,
) -> bool {
    let mut borrows = BorrowCollector {
        cold_blocks,
        shared_borrows: IndexVec::from_elem(Vec::new(), &body.local_decls),
        address_taken: DenseBitSet::new_empty(body.local_decls.len()),
        used_on_hot_path: DenseBitSet::new_empty(body.local_decls.len()),
    };
    // Arguments are live on entry, which is never cold.
    for arg in body.args_iter() {
        borrows.used_on_hot_path.insert(arg);
    }
    borrows.visit_body(body);

    let typing_env = body.typing_env(tcx);
    let can_return = blocks_that_can_return(body);
    let mut reachable: IndexVec<BasicBlock, Option<DenseBitSet<BasicBlock>>> =
        IndexVec::from_elem(None, &body.basic_blocks);

    let mut rewrites = Vec::new();
    for (local, sites) in borrows.shared_borrows.iter_enumerated() {
        if sites.is_empty()
            || local == RETURN_PLACE
            || borrows.address_taken.contains(local)
            || !borrows.used_on_hot_path.contains(local)
        {
            continue;
        }
        let ty = body.local_decls[local].ty;
        if !tcx.type_is_copy_modulo_regions(typing_env, ty)
            || !ty.is_freeze(tcx, typing_env)
            || !is_padding_free_scalar_or_pair(tcx, typing_env, ty)
        {
            continue;
        }
        let in_diverging_cold_blocks = sites.iter().enumerate().all(|(i, site)| {
            let bb = site.block;
            cold_blocks[bb]
                && !body.basic_blocks[bb].is_cleanup
                && !can_return.contains(bb)
                && sites[..i].iter().all(|earlier| earlier.block != bb)
        });
        // If one borrow could run after another (or after itself), two pointers that used to be
        // equal would point to different copies.
        let at_most_one_per_call = in_diverging_cold_blocks
            && sites.iter().all(|site| {
                let reached = reachable[site.block]
                    .get_or_insert_with(|| blocks_reachable_from(body, site.block));
                sites.iter().all(|other| !reached.contains(other.block))
            });
        if at_most_one_per_call {
            rewrites.push(local);
        }
    }

    if rewrites.is_empty() {
        return false;
    }

    let mut edits = Vec::new();
    for local in rewrites {
        let sites = &borrows.shared_borrows[local];
        let ty = body.local_decls[local].ty;
        let source_info = *body.source_info(sites[0]);
        let copy = body.local_decls.push(LocalDecl::with_source_info(ty, source_info));
        edits.extend(sites.iter().map(|&site| (site, local, copy)));
    }
    // Insert back to front so that the statement indices of pending edits stay valid.
    edits.sort_unstable_by_key(|&(site, ..)| Reverse((site.block, site.statement_index)));

    // Only statements change, so the CFG caches stay valid.
    let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
    for (site, local, copy) in edits {
        let statements = &mut basic_blocks[site.block].statements;
        let source_info = statements[site.statement_index].source_info;
        let StatementKind::Assign(assign) = &mut statements[site.statement_index].kind else {
            bug!("expected the borrow of {local:?} at {site:?} to be an assignment");
        };
        let (_, Rvalue::Ref(_, BorrowKind::Shared, borrowed)) = &mut **assign else {
            bug!("expected a shared borrow of {local:?} at {site:?}");
        };
        *borrowed = Place::from(copy);
        statements.splice(
            site.statement_index..site.statement_index,
            [
                Statement::new(source_info, StatementKind::StorageLive(copy)),
                Statement::new(
                    source_info,
                    StatementKind::Assign(Box::new((
                        Place::from(copy),
                        Rvalue::Use(Operand::Copy(Place::from(local)), WithRetag::No),
                    ))),
                ),
            ],
        );
    }

    true
}

/// Per local: the shared borrows of the bare local (rewrite candidates), whether its address can be
/// observed any other way, and whether it is used outside cold blocks.
struct BorrowCollector<'a> {
    cold_blocks: &'a IndexVec<BasicBlock, bool>,
    shared_borrows: IndexVec<Local, Vec<Location>>,
    address_taken: DenseBitSet<Local>,
    used_on_hot_path: DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for BorrowCollector<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Going through a `Deref` exposes the pointee's address, not the local's.
        if context.may_observe_address() && !place.is_indirect() {
            if place.projection.is_empty()
                && context == PlaceContext::NonMutatingUse(NonMutatingUseContext::SharedBorrow)
            {
                self.shared_borrows[place.local].push(location);
            } else {
                self.address_taken.insert(place.local);
            }
        }
        self.super_place(place, context, location);
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        if context.is_use() && !self.cold_blocks[location.block] {
            self.used_on_hot_path.insert(local);
        }
    }
}

/// Whether values of `ty` are one or two scalars without padding. Copying such a value preserves
/// all of its bytes, and these are the types codegen can keep in registers when nothing borrows
/// them.
fn is_padding_free_scalar_or_pair<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else { return false };
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => scalar.size(&tcx) == layout.size,
        BackendRepr::ScalarPair { a, b, b_offset } => {
            a.size(&tcx) == b_offset && b_offset + b.size(&tcx) == layout.size
        }
        _ => false,
    }
}

/// Blocks from which a `return` (or another way of leaving the function normally) is reachable.
fn blocks_that_can_return(body: &Body<'_>) -> DenseBitSet<BasicBlock> {
    let mut can_return = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut worklist = Vec::new();
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        if matches!(
            data.terminator().kind,
            TerminatorKind::Return
                | TerminatorKind::TailCall { .. }
                | TerminatorKind::Yield { .. }
                | TerminatorKind::CoroutineDrop
        ) && can_return.insert(bb)
        {
            worklist.push(bb);
        }
    }
    let predecessors = body.basic_blocks.predecessors();
    while let Some(bb) = worklist.pop() {
        for &pred in &predecessors[bb] {
            if can_return.insert(pred) {
                worklist.push(pred);
            }
        }
    }
    can_return
}

/// Blocks reachable from `start` through normal (non-unwind) edges. `start` itself is included
/// only if it lies on a cycle.
fn blocks_reachable_from(body: &Body<'_>, start: BasicBlock) -> DenseBitSet<BasicBlock> {
    let mut reached = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut worklist = vec![start];
    while let Some(bb) = worklist.pop() {
        for succ in body.basic_blocks[bb].terminator().successors() {
            if !body.basic_blocks[succ].is_cleanup && reached.insert(succ) {
                worklist.push(succ);
            }
        }
    }
    reached
}
