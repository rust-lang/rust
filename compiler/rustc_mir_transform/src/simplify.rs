//! A number of passes which remove various redundancies in the CFG.
//!
//! The `SimplifyCfg` pass gets rid of unnecessary blocks in the CFG, whereas the `SimplifyLocals`
//! gets rid of all the unnecessary local variable declarations.
//!
//! The `SimplifyLocals` pass is kinda expensive and therefore not very suitable to be run often.
//! Most of the passes should not care or be impacted in meaningful ways due to extra locals
//! either, so running the pass once, right before codegen, should suffice.
//!
//! On the other side of the spectrum, the `SimplifyCfg` pass is considerably cheap to run, thus
//! one should run it after every pass which may modify CFG in significant ways. This pass must
//! also be run before any analysis passes because it removes dead blocks, and some of these can be
//! ill-typed.
//!
//! The cause of this typing issue is typeck allowing most blocks whose end is not reachable have
//! an arbitrary return type, rather than having the usual () return type (as a note, typeck's
//! notion of reachability is in fact slightly weaker than MIR CFG reachability - see #31617). A
//! standard example of the situation is:
//!
//! ```rust
//!   fn example() {
//!       let _a: char = { return; };
//!   }
//! ```
//!
//! Here the block (`{ return; }`) has the return type `char`, rather than `()`, but the MIR we
//! naively generate still contains the `_a = ()` write in the unreachable block "after" the
//! return.
//!
//! **WARNING**: This is one of the few optimizations that runs on built and analysis MIR, and
//! so its effects may affect the type-checking, borrow-checking, and other analysis of MIR.
//! We must be extremely careful to only apply optimizations that preserve UB and all
//! non-determinism, since changes here can affect which programs compile in an insta-stable way.
//! The normal logic that a program with UB can be changed to do anything does not apply to
//! pre-"runtime" MIR!

use rustc_index::{Idx, IndexSlice, IndexVec};
use rustc_middle::mir::visit::{MutVisitor, MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_span::DUMMY_SP;
use smallvec::SmallVec;
use tracing::{debug, trace};

pub(super) enum SimplifyCfg {
    Initial,
    PromoteConsts,
    RemoveFalseEdges,
    /// Runs at the beginning of "analysis to runtime" lowering, *before* drop elaboration.
    PostAnalysis,
    /// Runs at the end of "analysis to runtime" lowering, *after* drop elaboration.
    /// This is before the main optimization passes on runtime MIR kick in.
    PreOptimizations,
    Final,
    MakeShim,
    AfterUnreachableEnumBranching,
}

impl SimplifyCfg {
    fn name(&self) -> &'static str {
        match self {
            SimplifyCfg::Initial => "SimplifyCfg-initial",
            SimplifyCfg::PromoteConsts => "SimplifyCfg-promote-consts",
            SimplifyCfg::RemoveFalseEdges => "SimplifyCfg-remove-false-edges",
            SimplifyCfg::PostAnalysis => "SimplifyCfg-post-analysis",
            SimplifyCfg::PreOptimizations => "SimplifyCfg-pre-optimizations",
            SimplifyCfg::Final => "SimplifyCfg-final",
            SimplifyCfg::MakeShim => "SimplifyCfg-make_shim",
            SimplifyCfg::AfterUnreachableEnumBranching => {
                "SimplifyCfg-after-unreachable-enum-branching"
            }
        }
    }
}

pub(super) fn simplify_cfg<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    CfgSimplifier::new(tcx, body).simplify();
    remove_dead_blocks(body);

    // FIXME: Should probably be moved into some kind of pass manager
    body.basic_blocks_mut().raw.shrink_to_fit();
}

impl<'tcx> crate::MirPass<'tcx> for SimplifyCfg {
    fn name(&self) -> &'static str {
        self.name()
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("SimplifyCfg({:?}) - simplifying {:?}", self.name(), body.source);
        simplify_cfg(tcx, body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

struct CfgSimplifier<'a, 'tcx> {
    preserve_switch_reads: bool,
    basic_blocks: &'a mut IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
    pred_count: IndexVec<BasicBlock, u32>,
}

impl<'a, 'tcx> CfgSimplifier<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, body: &'a mut Body<'tcx>) -> Self {
        let mut pred_count = IndexVec::from_elem(0u32, &body.basic_blocks);

        // we can't use mir.predecessors() here because that counts
        // dead blocks, which we don't want to.
        pred_count[START_BLOCK] = 1;

        for (_, data) in traversal::preorder(body) {
            if let Some(ref term) = data.terminator {
                for tgt in term.successors() {
                    pred_count[tgt] += 1;
                }
            }
        }

        // Preserve `SwitchInt` reads on built and analysis MIR, or if `-Zmir-preserve-ub`.
        let preserve_switch_reads = matches!(body.phase, MirPhase::Built | MirPhase::Analysis(_))
            || tcx.sess.opts.unstable_opts.mir_preserve_ub;
        let basic_blocks = body.basic_blocks_mut();

        CfgSimplifier { preserve_switch_reads, basic_blocks, pred_count }
    }

    fn simplify(mut self) {
        self.strip_nops();

        // Vec of the blocks that should be merged. We store the indices here, instead of the
        // statements itself to avoid moving the (relatively) large statements twice.
        // We do not push the statements directly into the target block (`bb`) as that is slower
        // due to additional reallocations
        let mut merged_blocks = Vec::new();
        loop {
            let mut changed = false;

            for bb in self.basic_blocks.indices() {
                if self.pred_count[bb] == 0 {
                    continue;
                }

                debug!("simplifying {:?}", bb);

                let mut terminator =
                    self.basic_blocks[bb].terminator.take().expect("invalid terminator state");

                for successor in terminator.successors_mut() {
                    self.collapse_goto_chain(successor, &mut changed);
                }

                let mut inner_changed = true;
                merged_blocks.clear();
                while inner_changed {
                    inner_changed = false;
                    inner_changed |= self.simplify_branch(&mut terminator);
                    inner_changed |= self.merge_successor(&mut merged_blocks, &mut terminator);
                    changed |= inner_changed;
                }

                let statements_to_merge =
                    merged_blocks.iter().map(|&i| self.basic_blocks[i].statements.len()).sum();

                if statements_to_merge > 0 {
                    let mut statements = std::mem::take(&mut self.basic_blocks[bb].statements);
                    statements.reserve(statements_to_merge);
                    for &from in &merged_blocks {
                        statements.append(&mut self.basic_blocks[from].statements);
                    }
                    self.basic_blocks[bb].statements = statements;
                }

                self.basic_blocks[bb].terminator = Some(terminator);
            }

            if !changed {
                break;
            }
        }
    }

    /// This function will return `None` if
    /// * the block has statements
    /// * the block has a terminator other than `goto`
    /// * the block has no terminator (meaning some other part of the current optimization stole it)
    fn take_terminator_if_simple_goto(&mut self, bb: BasicBlock) -> Option<Terminator<'tcx>> {
        match self.basic_blocks[bb] {
            BasicBlockData {
                ref statements,
                terminator:
                    ref mut terminator @ Some(Terminator { kind: TerminatorKind::Goto { .. }, .. }),
                ..
            } if statements.is_empty() => terminator.take(),
            // if `terminator` is None, this means we are in a loop. In that
            // case, let all the loop collapse to its entry.
            _ => None,
        }
    }

    /// Collapse a goto chain starting from `start`
    fn collapse_goto_chain(&mut self, start: &mut BasicBlock, changed: &mut bool) {
        // Using `SmallVec` here, because in some logs on libcore oli-obk saw many single-element
        // goto chains. We should probably benchmark different sizes.
        let mut terminators: SmallVec<[_; 1]> = Default::default();
        let mut current = *start;
        while let Some(terminator) = self.take_terminator_if_simple_goto(current) {
            let Terminator { kind: TerminatorKind::Goto { target }, .. } = terminator else {
                unreachable!();
            };
            terminators.push((current, terminator));
            current = target;
        }
        let last = current;
        *start = last;
        while let Some((current, mut terminator)) = terminators.pop() {
            let Terminator { kind: TerminatorKind::Goto { ref mut target }, .. } = terminator
            else {
                unreachable!();
            };
            *changed |= *target != last;
            *target = last;
            debug!("collapsing goto chain from {:?} to {:?}", current, target);

            if self.pred_count[current] == 1 {
                // This is the last reference to current, so the pred-count to
                // to target is moved into the current block.
                self.pred_count[current] = 0;
            } else {
                self.pred_count[*target] += 1;
                self.pred_count[current] -= 1;
            }
            self.basic_blocks[current].terminator = Some(terminator);
        }
    }

    // merge a block with 1 `goto` predecessor to its parent
    fn merge_successor(
        &mut self,
        merged_blocks: &mut Vec<BasicBlock>,
        terminator: &mut Terminator<'tcx>,
    ) -> bool {
        let target = match terminator.kind {
            TerminatorKind::Goto { target } if self.pred_count[target] == 1 => target,
            _ => return false,
        };

        debug!("merging block {:?} into {:?}", target, terminator);
        *terminator = match self.basic_blocks[target].terminator.take() {
            Some(terminator) => terminator,
            None => {
                // unreachable loop - this should not be possible, as we
                // don't strand blocks, but handle it correctly.
                return false;
            }
        };

        merged_blocks.push(target);
        self.pred_count[target] = 0;

        true
    }

    // turn a branch with all successors identical to a goto
    fn simplify_branch(&mut self, terminator: &mut Terminator<'tcx>) -> bool {
        // Removing a `SwitchInt` terminator may remove reads that result in UB,
        // so we must not apply this optimization before borrowck or when
        // `-Zmir-preserve-ub` is set.
        if self.preserve_switch_reads {
            return false;
        }

        let TerminatorKind::SwitchInt { .. } = terminator.kind else {
            return false;
        };

        let first_succ = {
            if let Some(first_succ) = terminator.successors().next() {
                if terminator.successors().all(|s| s == first_succ) {
                    let count = terminator.successors().count();
                    self.pred_count[first_succ] -= (count - 1) as u32;
                    first_succ
                } else {
                    return false;
                }
            } else {
                return false;
            }
        };

        debug!("simplifying branch {:?}", terminator);
        terminator.kind = TerminatorKind::Goto { target: first_succ };
        true
    }

    fn strip_nops(&mut self) {
        for blk in self.basic_blocks.iter_mut() {
            blk.statements.retain(|stmt| !matches!(stmt.kind, StatementKind::Nop))
        }
    }
}

pub(super) fn simplify_duplicate_switch_targets(terminator: &mut Terminator<'_>) {
    if let TerminatorKind::SwitchInt { targets, .. } = &mut terminator.kind {
        let otherwise = targets.otherwise();
        if targets.iter().any(|t| t.1 == otherwise) {
            *targets = SwitchTargets::new(
                targets.iter().filter(|t| t.1 != otherwise),
                targets.otherwise(),
            );
        }
    }
}

pub(super) fn remove_dead_blocks(body: &mut Body<'_>) {
    let should_deduplicate_unreachable = |bbdata: &BasicBlockData<'_>| {
        // CfgSimplifier::simplify leaves behind some unreachable basic blocks without a
        // terminator. Those blocks will be deleted by remove_dead_blocks, but we run just
        // before then so we need to handle missing terminators.
        // We also need to prevent confusing cleanup and non-cleanup blocks. In practice we
        // don't emit empty unreachable cleanup blocks, so this simple check suffices.
        bbdata.terminator.is_some() && bbdata.is_empty_unreachable() && !bbdata.is_cleanup
    };

    let reachable = traversal::reachable_as_bitset(body);
    let empty_unreachable_blocks = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(bb, bbdata)| should_deduplicate_unreachable(bbdata) && reachable.contains(*bb))
        .count();

    let num_blocks = body.basic_blocks.len();
    if num_blocks == reachable.count() && empty_unreachable_blocks <= 1 {
        return;
    }

    let basic_blocks = body.basic_blocks.as_mut();

    let mut replacements: Vec<_> = (0..num_blocks).map(BasicBlock::new).collect();
    let mut orig_index = 0;
    let mut used_index = 0;
    let mut kept_unreachable = None;
    let mut deduplicated_unreachable = false;
    basic_blocks.raw.retain(|bbdata| {
        let orig_bb = BasicBlock::new(orig_index);
        if !reachable.contains(orig_bb) {
            orig_index += 1;
            return false;
        }

        let used_bb = BasicBlock::new(used_index);
        if should_deduplicate_unreachable(bbdata) {
            let kept_unreachable = *kept_unreachable.get_or_insert(used_bb);
            if kept_unreachable != used_bb {
                replacements[orig_index] = kept_unreachable;
                deduplicated_unreachable = true;
                orig_index += 1;
                return false;
            }
        }

        replacements[orig_index] = used_bb;
        used_index += 1;
        orig_index += 1;
        true
    });

    // If we deduplicated unreachable blocks we erase their source_info as we
    // can no longer attribute their code to a particular location in the
    // source.
    if deduplicated_unreachable {
        basic_blocks[kept_unreachable.unwrap()].terminator_mut().source_info =
            SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE };
    }

    for block in basic_blocks {
        for target in block.terminator_mut().successors_mut() {
            *target = replacements[target.index()];
        }
    }
}

pub(super) enum SimplifyLocals {
    BeforeConstProp,
    AfterGVN,
    Final,
}

impl<'tcx> crate::MirPass<'tcx> for SimplifyLocals {
    fn name(&self) -> &'static str {
        match &self {
            SimplifyLocals::BeforeConstProp => "SimplifyLocals-before-const-prop",
            SimplifyLocals::AfterGVN => "SimplifyLocals-after-value-numbering",
            SimplifyLocals::Final => "SimplifyLocals-final",
        }
    }

    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("running SimplifyLocals on {:?}", body.source);

        // First, we're going to get a count of *actual* uses for every `Local`.
        let mut used_locals = UsedLocals::new(body);

        // Next, we're going to remove any `Local` with zero actual uses. When we remove those
        // `Locals`, we're also going to subtract any uses of other `Locals` from the `used_locals`
        // count. For example, if we removed `_2 = discriminant(_1)`, then we'll subtract one from
        // `use_counts[_1]`. That in turn might make `_1` unused, so we loop until we hit a
        // fixedpoint where there are no more unused locals.
        remove_unused_definitions_helper(&mut used_locals, body);

        // Finally, we'll actually do the work of shrinking `body.local_decls` and remapping the
        // `Local`s.
        let map = make_local_map(&mut body.local_decls, &used_locals);

        // Only bother running the `LocalUpdater` if we actually found locals to remove.
        if map.iter().any(Option::is_none) {
            // Update references to all vars and tmps now
            let mut updater = LocalUpdater { map, tcx };
            updater.visit_body_preserves_cfg(body);

            body.local_decls.shrink_to_fit();
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

pub(super) fn remove_unused_definitions<'tcx>(body: &mut Body<'tcx>) {
    // First, we're going to get a count of *actual* uses for every `Local`.
    let mut used_locals = UsedLocals::new(body);

    // Next, we're going to remove any `Local` with zero actual uses. When we remove those
    // `Locals`, we're also going to subtract any uses of other `Locals` from the `used_locals`
    // count. For example, if we removed `_2 = discriminant(_1)`, then we'll subtract one from
    // `use_counts[_1]`. That in turn might make `_1` unused, so we loop until we hit a
    // fixedpoint where there are no more unused locals.
    remove_unused_definitions_helper(&mut used_locals, body);
}

/// Construct the mapping while swapping out unused stuff out from the `vec`.
fn make_local_map<V>(
    local_decls: &mut IndexVec<Local, V>,
    used_locals: &UsedLocals,
) -> IndexVec<Local, Option<Local>> {
    let mut map: IndexVec<Local, Option<Local>> = IndexVec::from_elem(None, local_decls);
    let mut used = Local::ZERO;

    for alive_index in local_decls.indices() {
        // `is_used` treats the `RETURN_PLACE` and arguments as used.
        if !used_locals.is_used(alive_index) {
            continue;
        }

        map[alive_index] = Some(used);
        if alive_index != used {
            local_decls.swap(alive_index, used);
        }
        used.increment_by(1);
    }
    local_decls.truncate(used.index());
    map
}

/// Keeps track of used & unused locals.
struct UsedLocals {
    increment: bool,
    arg_count: u32,
    use_count: IndexVec<Local, u32>,
}

impl UsedLocals {
    /// Determines which locals are used & unused in the given body.
    fn new(body: &Body<'_>) -> Self {
        let mut this = Self {
            increment: true,
            arg_count: body.arg_count.try_into().unwrap(),
            use_count: IndexVec::from_elem(0, &body.local_decls),
        };
        this.visit_body(body);
        this
    }

    /// Checks if local is used.
    ///
    /// Return place and arguments are always considered used.
    fn is_used(&self, local: Local) -> bool {
        trace!("is_used({:?}): use_count: {:?}", local, self.use_count[local]);
        local.as_u32() <= self.arg_count || self.use_count[local] != 0
    }

    /// Updates the use counts to reflect the removal of given statement.
    fn statement_removed(&mut self, statement: &Statement<'_>) {
        self.increment = false;

        // The location of the statement is irrelevant.
        let location = Location::START;
        self.visit_statement(statement, location);
    }

    /// Visits a left-hand side of an assignment.
    fn visit_lhs(&mut self, place: &Place<'_>, location: Location) {
        if place.is_indirect() {
            // A use, not a definition.
            self.visit_place(place, PlaceContext::MutatingUse(MutatingUseContext::Store), location);
        } else {
            // A definition. The base local itself is not visited, so this occurrence is not counted
            // toward its use count. There might be other locals still, used in an indexing
            // projection.
            self.super_projection(
                place.as_ref(),
                PlaceContext::MutatingUse(MutatingUseContext::Projection),
                location,
            );
        }
    }
}

impl<'tcx> Visitor<'tcx> for UsedLocals {
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::Intrinsic(..)
            | StatementKind::Retag(..)
            | StatementKind::Coverage(..)
            | StatementKind::FakeRead(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::AscribeUserType(..) => {
                self.super_statement(statement, location);
            }

            StatementKind::ConstEvalCounter | StatementKind::Nop => {}

            StatementKind::StorageLive(_local) | StatementKind::StorageDead(_local) => {}

            StatementKind::Assign(box (ref place, ref rvalue)) => {
                if rvalue.is_safe_to_remove() {
                    self.visit_lhs(place, location);
                    self.visit_rvalue(rvalue, location);
                } else {
                    self.super_statement(statement, location);
                }
            }

            StatementKind::SetDiscriminant { ref place, variant_index: _ }
            | StatementKind::Deinit(ref place)
            | StatementKind::BackwardIncompatibleDropHint { ref place, reason: _ } => {
                self.visit_lhs(place, location);
            }
        }
    }

    fn visit_local(&mut self, local: Local, _ctx: PlaceContext, _location: Location) {
        if self.increment {
            self.use_count[local] += 1;
        } else {
            assert_ne!(self.use_count[local], 0);
            self.use_count[local] -= 1;
        }
    }
}

/// Removes unused definitions. Updates the used locals to reflect the changes made.
fn remove_unused_definitions_helper(used_locals: &mut UsedLocals, body: &mut Body<'_>) {
    // The use counts are updated as we remove the statements. A local might become unused
    // during the retain operation, leading to a temporary inconsistency (storage statements or
    // definitions referencing the local might remain). For correctness it is crucial that this
    // computation reaches a fixed point.

    let mut modified = true;
    while modified {
        modified = false;

        for data in body.basic_blocks.as_mut_preserves_cfg() {
            // Remove unnecessary StorageLive and StorageDead annotations.
            data.statements.retain(|statement| {
                let keep = match &statement.kind {
                    StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                        used_locals.is_used(*local)
                    }
                    StatementKind::Assign(box (place, _)) => used_locals.is_used(place.local),

                    StatementKind::SetDiscriminant { place, .. }
                    | StatementKind::BackwardIncompatibleDropHint { place, reason: _ }
                    | StatementKind::Deinit(place) => used_locals.is_used(place.local),
                    StatementKind::Nop => false,
                    _ => true,
                };

                if !keep {
                    trace!("removing statement {:?}", statement);
                    modified = true;
                    used_locals.statement_removed(statement);
                }

                keep
            });
        }
    }
}

struct LocalUpdater<'tcx> {
    map: IndexVec<Local, Option<Local>>,
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for LocalUpdater<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, l: &mut Local, _: PlaceContext, _: Location) {
        *l = self.map[*l].unwrap();
    }
}
