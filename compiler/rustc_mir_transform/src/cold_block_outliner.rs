//! Experimental MIR pass (`ColdBlockOutliner`) that prepares cold panic, error, and caught-unwind
//! drop regions for outlining and deduplication across generic instantiations.
//!
//! Together with `KnownVariantDrops` and pre-inliner hot/cold splitting, this pass addresses both:
//! - `c8_coldpayload`: caught-panic payload drops (`Box<dyn Any + Send>`) on the nonzero path of
//!   `catch_unwind`, and
//! - `C10`: cold `panic!` / `assert!` formatting blocks and `?` (`FromResidual::from_residual`)
//!   error paths inside generic or `#[inline]` functions.
//!
//! Specifically, `ColdBlockOutliner`:
//! 1. Computes cold basic blocks (`find_cold_blocks`), marking `Unreachable` blocks, calls to
//!    `#[cold]` functions, caught-panic paths after `catch_unwind`, and `FromResidual::from_residual`
//!    (`?` break) paths, then propagating coldness backwards through the CFG.
//! 2. Sinks pure cold-only setup statements out of hot predecessor blocks into cold entry blocks so
//!    they do not bloat the hot path or increase the live-in parameter count of outlined regions.
//! 3. Copies multi-block `Copy` locals into cold-local temporaries before taking shared references
//!    in cold blocks so that `non_ssa_locals` keeps the hot local in an SSA register.
//!
//! Enabled via `-Zmir-enable-passes=+ColdBlockOutliner` or `RUSTC_EXP_COLD_OUTLINE=1`.

use std::sync::LazyLock;

use rustc_attr_ir::lang_items::LangItem;
use rustc_index::IndexVec;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::sym;
use tracing::debug;

use crate::PassPolicy;

static COLD_OUTLINE: LazyLock<bool> =
    LazyLock::new(|| std::env::var("RUSTC_EXP_COLD_OUTLINE").is_ok_and(|v| v == "1"));

static COLD_CATCH_UNWIND: LazyLock<bool> =
    LazyLock::new(|| std::env::var("RUSTC_EXP_COLD_CATCH_UNWIND").is_ok_and(|v| v == "1"));

pub(super) struct ColdBlockOutliner;

pub(crate) fn is_enabled(tcx: TyCtxt<'_>) -> bool {
    if *COLD_OUTLINE {
        return true;
    }
    tcx.sess
        .opts
        .unstable_opts
        .mir_enable_passes
        .iter()
        .rev()
        .find(|(name, _)| name == "ColdBlockOutliner")
        .is_some_and(|(_, enabled)| *enabled)
}

impl<'tcx> crate::MirPass<'tcx> for ColdBlockOutliner {
    fn policy(&self, _ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(*COLD_OUTLINE)
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

        let mut changed = false;
        changed |= sink_cold_only_statements(body, &cold_blocks);
        changed |= decouple_cold_borrows(tcx, body, &cold_blocks);

        if changed {
            debug!(?def_id, "ColdBlockOutliner modified MIR body");
            body.basic_blocks.invalidate_cfg_cache();
        }
    }
}

/// Marks the successors of `switchInt(catch_unwind(..))` that are taken for a nonzero result
/// (a caught panic) as cold.
fn mark_caught_panic_blocks_cold<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir: &Body<'tcx>,
    cold_blocks: &mut IndexVec<BasicBlock, bool>,
) {
    for bb_data in mir.basic_blocks.iter() {
        let TerminatorKind::Call { ref func, destination, target: Some(target), .. } =
            bb_data.terminator().kind
        else {
            continue;
        };
        let ty::FnDef(def_id, _) = *func.ty(&mir.local_decls, tcx).kind() else { continue };
        if !tcx.is_intrinsic(def_id, sym::catch_unwind) {
            continue;
        }
        let Some(result) = destination.as_local() else { continue };
        let target_data = &mir.basic_blocks[target];
        let TerminatorKind::SwitchInt { ref discr, ref targets } = target_data.terminator().kind
        else {
            continue;
        };
        if discr.place().and_then(|place| place.as_local()) != Some(result) {
            continue;
        }
        let result_rewritten =
            target_data.statements.iter().any(|statement| match &statement.kind {
                StatementKind::Assign(assign) => assign.0.local == result,
                StatementKind::SetDiscriminant { place, .. } => place.local == result,
                StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                    *local == result
                }
                _ => false,
            });
        if result_rewritten {
            continue;
        }
        let no_panic = targets.target_for_value(0);
        for (value, bb) in targets.iter() {
            if value != 0 && bb != no_panic {
                cold_blocks[bb] = true;
            }
        }
        let otherwise = targets.otherwise();
        if targets.iter().any(|(value, _)| value == 0) && otherwise != no_panic {
            cold_blocks[otherwise] = true;
        }
    }
}

pub(crate) fn is_try_from_residual(tcx: TyCtxt<'_>, def_id: rustc_hir::def_id::DefId) -> bool {
    tcx.is_lang_item(def_id, LangItem::TryTraitFromResidual)
        || tcx
            .trait_item_of(def_id)
            .is_some_and(|ti| tcx.is_lang_item(ti, LangItem::TryTraitFromResidual))
}

/// Identifies cold basic blocks in `mir`, including blocks ending in `Unreachable`, calls to
/// `#[cold]` functions, `FromResidual::from_residual` (`?` residual) calls, and caught-panic
/// arms of `catch_unwind`, then propagates coldness backwards through postorder traversal.
pub(crate) fn find_cold_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    mir: &Body<'tcx>,
) -> IndexVec<BasicBlock, bool> {
    let local_decls = &mir.local_decls;
    let mut cold_blocks: IndexVec<BasicBlock, bool> = IndexVec::from_elem(false, &mir.basic_blocks);

    if *COLD_CATCH_UNWIND || is_enabled(tcx) {
        mark_caught_panic_blocks_cold(tcx, mir, &mut cold_blocks);
    }

    let mark_residual_cold = is_enabled(tcx);

    for (bb, bb_data) in traversal::postorder(mir) {
        if bb_data.is_cleanup {
            continue;
        }
        let terminator = bb_data.terminator();

        match terminator.kind {
            TerminatorKind::Call { ref func, .. } | TerminatorKind::TailCall { ref func, .. }
                if let ty::FnDef(def_id, ..) = *func.ty(local_decls, tcx).kind() =>
            {
                let attrs = tcx.codegen_fn_attrs(def_id);
                if attrs.flags.contains(CodegenFnAttrFlags::COLD)
                    || (mark_residual_cold && is_try_from_residual(tcx, def_id))
                {
                    cold_blocks[bb] = true;
                    continue;
                }
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

/// Counts uses and definitions of each `Local` across the MIR body.
struct LocalUseCounts {
    uses_outside_block: IndexVec<Local, usize>,
    uses_in_target_block: IndexVec<Local, usize>,
    target_block: BasicBlock,
    source_block: BasicBlock,
}

impl<'tcx> Visitor<'tcx> for LocalUseCounts {
    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        if context.is_storage_marker() {
            return;
        }
        if location.block == self.target_block {
            self.uses_in_target_block[local] += 1;
        } else if location.block != self.source_block {
            self.uses_outside_block[local] += 1;
        }
    }
}

fn is_pure_sinkable_rvalue<'tcx>(rvalue: &Rvalue<'tcx>) -> bool {
    matches!(
        rvalue,
        Rvalue::Use(..)
            | Rvalue::Ref(_, BorrowKind::Shared, _)
            | Rvalue::RawPtr(_, _)
            | Rvalue::Cast(_, _, _)
            | Rvalue::BinaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::Aggregate(_, _)
    )
}

fn rvalue_reads_local<'tcx>(rvalue: &Rvalue<'tcx>, target_local: Local) -> bool {
    struct ReadsLocal {
        target_local: Local,
        found: bool,
    }
    impl<'tcx> Visitor<'tcx> for ReadsLocal {
        fn visit_local(&mut self, local: Local, _context: PlaceContext, _location: Location) {
            if local == self.target_local {
                self.found = true;
            }
        }
    }
    let mut finder = ReadsLocal { target_local, found: false };
    finder.visit_rvalue(rvalue, START_BLOCK.start_location());
    finder.found
}

/// Sinks pure temporary assignments out of hot blocks into a cold successor block when the
/// assigned local is only used inside that cold successor block and the cold successor has a
/// single predecessor.
fn sink_cold_only_statements<'tcx>(
    body: &mut Body<'tcx>,
    cold_blocks: &IndexVec<BasicBlock, bool>,
) -> bool {
    let mut pred_counts: IndexVec<BasicBlock, usize> = IndexVec::from_elem(0, &body.basic_blocks);
    for data in body.basic_blocks.iter() {
        for succ in data.terminator().successors() {
            pred_counts[succ] += 1;
        }
    }

    let mut candidates = Vec::new();
    for (bb, data) in body.basic_blocks.iter_enumerated() {
        if data.is_cleanup || cold_blocks[bb] || data.statements.is_empty() {
            continue;
        }
        let TerminatorKind::SwitchInt { ref discr, ref targets } = data.terminator().kind else {
            continue;
        };
        let discr_local = discr.place().and_then(|p| p.as_local());
        let mut cold_targets =
            targets.all_targets().iter().copied().filter(|&t| {
                cold_blocks[t] && !body.basic_blocks[t].is_cleanup && pred_counts[t] == 1
            });
        let Some(cold_target) = cold_targets.next() else {
            continue;
        };
        if cold_targets.next().is_some() {
            continue;
        }
        candidates.push((bb, cold_target, discr_local));
    }

    let mut changed = false;
    for (bb_hot, bb_cold, discr_local) in candidates {
        let mut counts = LocalUseCounts {
            uses_outside_block: IndexVec::from_elem(0, &body.local_decls),
            uses_in_target_block: IndexVec::from_elem(0, &body.local_decls),
            target_block: bb_cold,
            source_block: bb_hot,
        };
        counts.visit_body(body);

        let statements = &body.basic_blocks[bb_hot].statements;
        let mut to_sink_indices = Vec::new();
        for (idx, stmt) in statements.iter().enumerate() {
            let StatementKind::Assign(ref assign) = stmt.kind else {
                continue;
            };
            let (place, ref rvalue) = **assign;
            let Some(local) = place.as_local() else {
                continue;
            };
            if Some(local) == discr_local || local == RETURN_PLACE {
                continue;
            }
            if !is_pure_sinkable_rvalue(rvalue) {
                continue;
            }
            if counts.uses_outside_block[local] != 0 || counts.uses_in_target_block[local] == 0 {
                continue;
            }
            // Ensure no later statement in `bb_hot` reads `local` or writes any local read by `rvalue`.
            let used_or_clobbered_later =
                statements[idx + 1..].iter().any(|later| match &later.kind {
                    StatementKind::Assign(later_assign) => {
                        let (later_place, ref later_rvalue) = **later_assign;
                        rvalue_reads_local(later_rvalue, local)
                            || later_place.local == local
                            || rvalue_reads_local(rvalue, later_place.local)
                    }
                    StatementKind::SetDiscriminant { place: later_place, .. } => {
                        later_place.local == local || rvalue_reads_local(rvalue, later_place.local)
                    }
                    StatementKind::StorageLive(l) | StatementKind::StorageDead(l) => {
                        *l == local || rvalue_reads_local(rvalue, *l)
                    }
                    _ => true,
                });
            if !used_or_clobbered_later {
                to_sink_indices.push(idx);
            }
        }

        if to_sink_indices.is_empty() {
            continue;
        }

        let bbs = body.basic_blocks.as_mut();
        let mut sunken = Vec::with_capacity(to_sink_indices.len());
        for &idx in to_sink_indices.iter().rev() {
            sunken.push(bbs[bb_hot].statements.remove(idx));
        }
        sunken.reverse();
        sunken.append(&mut bbs[bb_cold].statements);
        bbs[bb_cold].statements = sunken;
        changed = true;
    }

    changed
}

/// Copies `Copy` locals that are live across non-cold blocks into fresh cold-local temporaries
/// before taking a shared reference (`&local`) inside a cold block (such as `panic!("{x}")`).
///
/// Without this transformation, `rustc_codegen_ssa::mir::analyze::non_ssa_locals` sees the shared
/// borrow of `x` in the cold block and forces `x` into a stack `alloca` in the function entry
/// block (`start`), emitting `alloca` + `store` on the hot path and preventing `CodeExtractor`
/// from sinking the `alloca` into the outlined `.cold` function.
fn decouple_cold_borrows<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    cold_blocks: &IndexVec<BasicBlock, bool>,
) -> bool {
    let num_locals = body.local_decls.len();
    let mut local_blocks: IndexVec<Local, Option<BasicBlock>> =
        IndexVec::from_elem_n(None, num_locals);
    let mut multi_block_locals = rustc_index::bit_set::DenseBitSet::new_empty(num_locals);

    for arg in 1..=body.arg_count {
        multi_block_locals.insert(Local::from_usize(arg));
    }
    multi_block_locals.insert(RETURN_PLACE);

    struct LocalBlockTracker<'a> {
        local_blocks: &'a mut IndexVec<Local, Option<BasicBlock>>,
        multi_block_locals: &'a mut rustc_index::bit_set::DenseBitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for LocalBlockTracker<'_> {
        fn visit_local(&mut self, local: Local, _context: PlaceContext, location: Location) {
            if self.multi_block_locals.contains(local) {
                return;
            }
            match self.local_blocks[local] {
                None => self.local_blocks[local] = Some(location.block),
                Some(prev_bb) if prev_bb != location.block => {
                    self.multi_block_locals.insert(local);
                }
                Some(_) => {}
            }
        }
    }

    LocalBlockTracker {
        local_blocks: &mut local_blocks,
        multi_block_locals: &mut multi_block_locals,
    }
    .visit_body(body);

    let typing_env = body.typing_env(tcx);
    let mut changed = false;

    for bb in body.basic_blocks.indices() {
        if !cold_blocks[bb] || body.basic_blocks[bb].is_cleanup {
            continue;
        }
        let stmt_len = body.basic_blocks[bb].statements.len();
        if stmt_len == 0 {
            continue;
        }

        let mut new_stmts = Vec::with_capacity(stmt_len);
        let old_stmts = std::mem::take(&mut body.basic_blocks.as_mut()[bb].statements);

        for mut stmt in old_stmts {
            let mut copy_prefix: Option<(Local, Local, SourceInfo)> = None;
            if let StatementKind::Assign(ref mut assign) = stmt.kind {
                let (_, ref mut rvalue) = **assign;
                if let Rvalue::Ref(region, BorrowKind::Shared, ref mut borrowed_place) = *rvalue
                    && let Some(local) = borrowed_place.as_local()
                    && local.as_usize() < num_locals
                    && multi_block_locals.contains(local)
                {
                    let ty = body.local_decls[local].ty;
                    if tcx.type_is_copy_modulo_regions(typing_env, ty) {
                        let source_info = stmt.source_info;
                        let cold_local =
                            body.local_decls.push(LocalDecl::with_source_info(ty, source_info));
                        *borrowed_place = Place::from(cold_local);
                        let _ = region;
                        copy_prefix = Some((cold_local, local, source_info));
                    }
                }
            }
            if let Some((cold_local, orig_local, source_info)) = copy_prefix {
                new_stmts.push(Statement::new(source_info, StatementKind::StorageLive(cold_local)));
                new_stmts.push(Statement::new(
                    source_info,
                    StatementKind::Assign(Box::new((
                        Place::from(cold_local),
                        Rvalue::Use(Operand::Copy(Place::from(orig_local)), WithRetag::No),
                    ))),
                ));
                changed = true;
            }
            new_stmts.push(stmt);
        }

        body.basic_blocks.as_mut()[bb].statements = new_stmts;
    }

    changed
}
