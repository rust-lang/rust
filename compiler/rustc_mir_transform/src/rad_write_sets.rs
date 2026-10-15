use either::Either;
use rustc_attr_ir::lang_items::LangItem;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir as hir;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    AwaitCheckpoints, BasicBlock, Body, BodyLoans, CheckpointObject, CheckpointTable, Depth,
    InstructionLoans, LoanId, Local, Location, Mutability, NonDivergingIntrinsic, Operand, Place,
    ProjectionElem, Rvalue, START_BLOCK, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{CAPTURE_STRUCT_LOCAL, TyCtxt, TypeVisitableExt};
use rustc_mir_dataflow::impls::{
    MaybeBorrowedLocals, MaybeLiveLocals, MaybeRequiresStorage, always_storage_live_locals,
};
use rustc_mir_dataflow::{Analysis as _, ResultsCursor};
use rustc_span::{Spanned, span_bug};

use crate::PassPolicy;
use crate::coroutine::CTX_ARG;
use crate::coroutine::layout::{LivenessInfo, locals_live_across_suspend_points};
use crate::rad_statement_ids::{id_at, id_label, repeated_ids, unnumbered};

pub(super) struct RadWriteSets;

impl<'tcx> crate::MirPass<'tcx> for RadWriteSets {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(ctx.opts.unstable_opts.rad_write_sets)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.source.promoted.is_some() {
            return;
        }
        let Some(plan) = analyze(tcx, body) else { return };
        // Keep what to copy at each `.await` for the pass inserting the checkpoints
        if let Some(coroutine) = body.coroutine.as_mut() {
            coroutine.rad_checkpoints = Some(plan.table());
        }
        with_no_trimmed_paths!(print(tcx, body, &plan));
    }
}

/// An `.await` has a poll loop, which rustc lowers as
/// `loop { match poll(..) { Ready(r) => break r, Pending => {} } cx = yield () }`,
/// and the code after the loop until the awaitee is dead
struct Await {
    /// The `Yield` that suspends while the awaited future is pending
    suspend: BasicBlock,
    /// The loop head, which the loop is entered through and the resume block jumps back to
    head: BasicBlock,
    /// The blocks that belong to the `.await` as a whole: the loop, and the code after its
    /// `Ready` exit until the awaitee's `StorageDead`
    blocks: DenseBitSet<BasicBlock>,
    /// The awaitee's `StorageDead` after the loop. The statements before it in that block still
    /// belong to the `.await`, the ones after it to the code that follows.
    ends: Vec<Location>,
}

/// Finds the poll loop and the end of the `.await` whose `Yield` is `suspend`
fn find_await(body: &Body<'_>, suspend: BasicBlock) -> Await {
    // Resume block of the Yield must end in `goto head`
    let unrecognised = |what| span_bug!(body.span, "`.await` at {suspend:?}: {what}");
    let TerminatorKind::Yield { resume, .. } = body[suspend].terminator().kind else {
        unrecognised("not a `Yield`")
    };
    let TerminatorKind::Goto { target: head } = body[resume].terminator().kind else {
        unrecognised("its resume block doesn't jump back to the loop head")
    };

    // The loop head pins the awaitee with `_p = &mut awaitee`
    let Some(awaitee) = body[head].statements.iter().find_map(|statement| match &statement.kind {
        StatementKind::Assign(assign) => match assign.1 {
            Rvalue::Ref(_, _, place) => place.as_local(),
            _ => None,
        },
        _ => None,
    }) else {
        unrecognised("no borrow of the awaitee in the loop head")
    };

    // The loop: every block on a path from the head to the `Yield`, and the resume block
    let mut blocks = DenseBitSet::new_empty(body.basic_blocks.len());
    blocks.insert(suspend);
    blocks.insert(resume);
    let mut stack = vec![suspend];
    while let Some(bb) = stack.pop() {
        if bb == head {
            continue;
        }
        for &pred in &body.basic_blocks.predecessors()[bb] {
            if blocks.insert(pred) {
                stack.push(pred);
            }
        }
    }

    // Walk down the CFG from the `Ready` exit until the awaitee is dead, skipping unwind paths and
    // unreachable edges.
    let normal = |bb: BasicBlock| {
        !body[bb].is_cleanup && !matches!(body[bb].terminator().kind, TerminatorKind::Unreachable)
    };
    let mut stack: Vec<BasicBlock> = blocks
        .iter()
        .filter(|&bb| bb != suspend)
        .flat_map(|bb| body[bb].terminator().successors())
        .filter(|&succ| !blocks.contains(succ) && normal(succ))
        .collect();

    let mut ends = Vec::new();
    while let Some(bb) = stack.pop() {
        // Find blocks where the await's future is dead, meaning the await is finished
        let dead = body[bb].statements.iter().position(|statement| {
            matches!(statement.kind, StatementKind::StorageDead(local) if local == awaitee)
        });
        if let Some(statement_index) = dead {
            let end = Location { block: bb, statement_index };
            if !ends.contains(&end) {
                ends.push(end);
            }
            continue;
        }

        // Record standard blocks that do not terminate the await section
        if !blocks.insert(bb) {
            continue;
        }

        if matches!(
            body[bb].terminator().kind,
            TerminatorKind::Yield { .. } | TerminatorKind::Return
        ) {
            unrecognised("no `StorageDead` of the awaitee after the loop")
        }

        stack.extend(body[bb].terminator().successors().filter(|&succ| normal(succ)));
    }
    Await { suspend, head, blocks, ends }
}

/// The locals live right after each `.await`'s `StorageDead`, by the rule
/// `locals_live_across_suspend_points` uses at a `Yield`
fn live_after_ends<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    movable: bool,
    awaits: &[Await],
) -> Vec<DenseBitSet<Local>> {
    let borrowed = MaybeBorrowedLocals.iterate_to_fixpoint(tcx, body, None);
    let requires_storage =
        MaybeRequiresStorage::new(body, &borrowed).iterate_to_fixpoint(tcx, body, None);
    let mut requires_storage = ResultsCursor::new_borrowing(body, &requires_storage);
    let mut borrowed = ResultsCursor::new_owning(body, borrowed);
    let mut liveness =
        MaybeLiveLocals.iterate_to_fixpoint(tcx, body, None).into_results_cursor(body);

    awaits
        .iter()
        .map(|awaited| {
            let mut live = DenseBitSet::new_empty(body.local_decls.len());
            for &end in &awaited.ends {
                // `StorageDead` isn't a use, so liveness is the same on both sides of it
                liveness.seek_after_primary_effect(end);
                let mut here = liveness.get().clone();
                if !movable {
                    borrowed.seek_after_primary_effect(end);
                    here.union(borrowed.get());
                }
                requires_storage.seek_after_primary_effect(end);
                here.intersect(requires_storage.get());
                live.union(&here);
            }
            live.remove(CAPTURE_STRUCT_LOCAL);
            // Each `poll` passes its own context, so a restore never needs an old one
            live.remove(CTX_ARG);
            live
        })
        .collect()
}

/// What the checkpoints of each `.await` of a body must save
struct CheckpointPlan<'tcx> {
    /// Polonius's results for the body
    loans: Option<&'tcx BodyLoans<'tcx>>,
    awaits: Vec<AwaitPlan<'tcx>>,
}

/// The two checkpoints of one `.await`
struct AwaitPlan<'tcx> {
    /// The block where the `.await` has a `Yield`
    suspend: BasicBlock,
    /// Taken where the poll loop is entered: what the code since the previous `.await` wrote
    before: WriteSet<'tcx>,
    /// The code of `before` starts at the function entry
    from_entry: bool,
    /// The earlier `.await`s whose end the code of `before` starts at
    after: Vec<usize>,
    /// Taken once the awaitee is dead: what the `.await` itself wrote
    effects: WriteSet<'tcx>,
    /// The awaitee's `StorageDead`, where `effects` is taken
    ends: Vec<Location>,
    /// Locals live where `effects` is taken that no `Yield` stores in the future yet
    pending: DenseBitSet<Local>,
}

/// The write set of one checkpoint: the writes in the code it covers and what it must copy
struct WriteSet<'tcx> {
    /// The locals stored in the future where the checkpoint is taken
    saved: DenseBitSet<Local>,
    /// The writes the liveness filter kept
    writes: Vec<Write<'tcx>>,
    /// Number of writes to locals that were filtered out
    not_saved: usize,
    /// What the checkpoint must copy, and whether we should follow references
    objects: FxIndexMap<CheckpointObject, Depth>,
}

impl CheckpointPlan<'_> {
    /// What to copy at each `.await`, without the details of how it was found
    fn table(&self) -> CheckpointTable {
        let list = |write_set: &WriteSet<'_>| -> Vec<_> {
            write_set.objects.iter().map(|(&object, &depth)| (object, depth)).collect()
        };
        let awaits = self
            .awaits
            .iter()
            .map(|plan| AwaitCheckpoints {
                before: list(&plan.before),
                effects: list(&plan.effects),
            })
            .collect();
        CheckpointTable { awaits }
    }
}

/// Records that the checkpoint copies `object` at least `depth` deep
fn record(
    objects: &mut FxIndexMap<CheckpointObject, Depth>,
    object: CheckpointObject,
    depth: Depth,
) {
    let entry = objects.entry(object).or_insert(depth);
    *entry = (*entry).max(depth);
}

/// An instruction in a checkpoint's code that writes memory the checkpoint may need
struct Write<'tcx> {
    location: Location,
    effect: Effect<'tcx>,
    /// The memory that the instruction can reach according to Polonius
    reach: Option<Reach<'tcx>>,
}

struct Reach<'tcx> {
    instruction: Option<&'tcx InstructionLoans>,
    targets: Targets<'tcx>,
}

/// Finds what the two checkpoints at each `.await` must save
fn analyze<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Option<CheckpointPlan<'tcx>> {
    // Every `Yield` of an async body is the suspension of an `.await`
    let awaits: Vec<Await> = body
        .basic_blocks
        .iter_enumerated()
        .filter(|(_, data)| matches!(data.terminator().kind, TerminatorKind::Yield { .. }))
        .map(|(bb, _)| find_await(body, bb))
        .collect();
    if awaits.is_empty() {
        return None;
    }

    let movable = tcx.coroutine_movability(body.source.def_id()) == hir::Movability::Movable;
    let liveness =
        locals_live_across_suspend_points(tcx, body, &always_storage_live_locals(body), movable);
    let live_after = live_after_ends(tcx, body, movable, &awaits);

    // Get live aliasing info from polonius
    let def = body.source.def_id().expect_local();
    let loans = tcx
        .mir_borrowck(tcx.typeck_root_def_id_local(def))
        .ok()
        .and_then(|result| result.loans.get(&def));

    let mut inside = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut end_of = FxIndexMap::default();
    for (index, awaited) in awaits.iter().enumerate() {
        inside.union(&awaited.blocks);
        for &end in &awaited.ends {
            end_of.insert(end.block, (index, end.statement_index));
        }
    }
    let analysis =
        Analysis { tcx, body, liveness, loans, defs: LocalDefs::new(tcx, body), inside, end_of };

    let awaits = awaits
        .iter()
        .zip(live_after)
        .enumerate()
        .map(|(index, (awaited, live))| analysis.await_plan(index, awaited, live))
        .collect();
    Some(CheckpointPlan { loans, awaits })
}

/// What `analyze` computes once per body, for every checkpoint to use
struct Analysis<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    liveness: LivenessInfo,
    loans: Option<&'tcx BodyLoans<'tcx>>,
    defs: LocalDefs,
    /// The blocks that belong to some `.await` as a whole
    inside: DenseBitSet<BasicBlock>,
    /// The blocks where an `.await` ends: its index and the statement of the awaitee's
    /// `StorageDead`
    end_of: FxIndexMap<BasicBlock, (usize, usize)>,
}

impl<'tcx> Analysis<'_, 'tcx> {
    /// Finds the two checkpoints of the `index`th `.await`, where `live` is what's live at its end
    fn await_plan(
        &self,
        index: usize,
        awaited: &Await,
        live: DenseBitSet<Local>,
    ) -> AwaitPlan<'tcx> {
        let body = self.body;

        // `before`: back from the loop's entry edges to the end of an earlier `.await` or the
        // function entry
        let mut code = Vec::new();
        let (mut from_entry, mut after) = (false, Vec::new());
        let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
        let mut stack: Vec<BasicBlock> = body.basic_blocks.predecessors()[awaited.head]
            .iter()
            .copied()
            .filter(|&pred| !awaited.blocks.contains(pred))
            .collect();
        while let Some(bb) = stack.pop() {
            if !seen.insert(bb) || self.inside.contains(bb) {
                continue;
            }
            let data = &body[bb];
            let first = match self.end_of.get(&bb) {
                // An earlier `.await` ends here, so only what follows its `StorageDead` counts
                Some(&(earlier, dead)) => {
                    if !after.contains(&earlier) {
                        after.push(earlier);
                    }
                    dead + 1
                }
                None => {
                    from_entry |= bb == START_BLOCK;
                    stack.extend(body.basic_blocks.predecessors()[bb].iter().copied());
                    0
                }
            };
            code.extend(
                (first..=data.statements.len())
                    .map(|statement_index| Location { block: bb, statement_index }),
            );
        }
        let before = self.write_set(&code, saved_at(body, &self.liveness, index));

        // `effects`: the loop without its `Yield`, and the code after it until the awaitee is dead
        let mut code = Vec::new();
        for bb in awaited.blocks.iter() {
            let len = body[bb].statements.len();
            let last = if bb == awaited.suspend { len } else { len + 1 };
            code.extend((0..last).map(|statement_index| Location { block: bb, statement_index }));
        }
        for end in &awaited.ends {
            code.extend(
                (0..end.statement_index)
                    .map(|statement_index| Location { block: end.block, statement_index }),
            );
        }
        // Only locals that some `Yield` stores have a slot in the future yet
        let mut saved = live.clone();
        let mut pending = live;
        for local in saved.clone().iter() {
            if self.liveness.saved_locals.get(local).is_some() {
                pending.remove(local);
            } else {
                saved.remove(local);
            }
        }
        let effects = self.write_set(&code, saved);

        AwaitPlan {
            suspend: awaited.suspend,
            before,
            from_entry,
            after,
            effects,
            ends: awaited.ends.clone(),
            pending,
        }
    }

    /// Finds what the checkpoint covering the instructions at `code` must copy, when `saved` are
    /// the locals stored in the future where it is taken
    fn write_set(&self, code: &[Location], saved: DenseBitSet<Local>) -> WriteSet<'tcx> {
        let body = self.body;
        // Find statements that modify objects
        let mut effects = Effects::default();
        let mut code = code.to_vec();
        code.sort();
        for location in code {
            match body.stmt_at(location) {
                Either::Left(statement) => effects.visit_statement(statement, location),
                Either::Right(terminator) => effects.visit_terminator(terminator, location),
            }
        }

        let mut objects = FxIndexMap::default();
        // Each checkpoint records the state its restore continues from
        record(&mut objects, CheckpointObject::Discriminant, Depth::Shallow);

        // Remove dead writes using liveness filter
        let mut writes = Vec::new();
        let mut not_saved = 0;
        for (location, effect) in effects.effects {
            // Find where it's actually writing
            if let Effect::Write(place, context) = effect
                && !place.is_indirect()
            {
                match self.object(place, &saved) {
                    Some(object) => record(&mut objects, object, Depth::Shallow),
                    // Drop glue can write through the references the dropped local holds, even
                    // when the local itself isn't needed after the checkpoint
                    None if context == MutatingUseContext::Drop => {}
                    // Drop writes to a temporary variable that aren't used after the checkpoint
                    None => {
                        not_saved += 1;
                        continue;
                    }
                }
            }
            // Find where this Place actually aliases
            let reach = if writes_through_pointers(&effect) {
                self.reach(location, &effect, &saved, &mut objects)
            } else {
                None
            };
            writes.push(Write { location, effect, reach });
        }
        WriteSet { saved, writes, not_saved, objects }
    }

    /// Find the saved local or upvar that holds `place` if it's stored in the future here
    fn object(&self, place: Place<'_>, saved: &DenseBitSet<Local>) -> Option<CheckpointObject> {
        if !in_future(place, saved) {
            return None;
        }
        if place.local == CAPTURE_STRUCT_LOCAL {
            let field = match place.projection.first() {
                Some(ProjectionElem::Field(field, _)) => Some(*field),
                _ => None,
            };
            return Some(CheckpointObject::Upvar(field));
        }
        Some(CheckpointObject::Saved(self.liveness.saved_locals.get(place.local)?))
    }

    /// Figure out which objects an instruction can write through its pointers
    fn reach(
        &self,
        location: Location,
        effect: &Effect<'tcx>,
        saved: &DenseBitSet<Local>,
        objects: &mut FxIndexMap<CheckpointObject, Depth>,
    ) -> Option<Reach<'tcx>> {
        // Without borrowck's results, the pointers can hold any saved local or memory
        let Some(loans) = self.loans.filter(|_| !matches!(effect, Effect::Intrinsic)) else {
            for local in saved.iter() {
                if let Some(object) = self.object(local.into(), saved) {
                    record(objects, object, Depth::Shallow);
                }
            }
            record(objects, CheckpointObject::Outside, Depth::Shallow);
            return None;
        };

        let instruction = id_at(self.body, location).and_then(|id| loans.instructions.get(&id));
        let targets = targets(self.tcx, self.body, loans, instruction);
        // Pointer points to objects in the future, a shallow copy
        for &place in &targets.places {
            if let Some(object) = self.object(place, saved) {
                record(objects, object, Depth::Shallow);
            }
        }
        // Pointer points to memory outside the filter
        if targets.outside {
            match self.holders(location, effect, saved) {
                Some(holders) => {
                    for holder in holders {
                        if let Some(object) = self.object(holder.into(), saved) {
                            record(objects, object, Depth::Deep);
                        }
                    }
                }
                None => record(objects, CheckpointObject::Outside, Depth::Shallow),
            }
        }
        Some(Reach { instruction, targets })
    }

    /// Find all the references an instruction writes through
    fn holders(
        &self,
        location: Location,
        effect: &Effect<'tcx>,
        saved: &DenseBitSet<Local>,
    ) -> Option<Vec<Local>> {
        match *effect {
            // A write through a pointer
            Effect::Write(
                place,
                MutatingUseContext::Store | MutatingUseContext::SetDiscriminant,
            ) => Some(self.pointer_holder(place, saved)?.slot().into_iter().collect()),
            // A call can write through any reference among its arguments, or its destination's
            Effect::Call => {
                let TerminatorKind::Call { args, destination, .. } =
                    &self.body[location.block].terminator().kind
                else {
                    return None;
                };
                let mut holders = Vec::new();
                for arg in args {
                    // Only types with lifetimes hold references (regions are erased here)
                    if arg.node.ty(self.body, self.tcx).has_erased_regions() {
                        holders.extend(self.operand_holder(&arg.node, saved, 0)?.slot());
                    }
                }
                if destination.is_indirect() {
                    holders.extend(self.pointer_holder(*destination, saved)?.slot());
                }
                Some(holders)
            }
            // Drops and everything else
            _ => None,
        }
    }

    /// The saved local holding the reference that `place` writes through
    fn pointer_holder(&self, place: Place<'tcx>, saved: &DenseBitSet<Local>) -> Option<Source> {
        if !single_deref(place) {
            return None;
        }
        self.holder(place.local, saved, 0)
    }

    /// Where the reference passed as `operand` leads
    fn operand_holder(
        &self,
        operand: &Operand<'tcx>,
        saved: &DenseBitSet<Local>,
        depth: usize,
    ) -> Option<Source> {
        match *operand {
            Operand::Copy(place) | Operand::Move(place) if !place.is_indirect() => {
                self.holder(place.local, saved, depth)
            }
            // A constant, or a reference read out of memory, which may have changed since
            _ => None,
        }
    }

    /// Follow the reference back to where it was originally stored
    fn holder(&self, local: Local, saved: &DenseBitSet<Local>, depth: usize) -> Option<Source> {
        // A saved local that can't change before the checkpoint still holds that reference then
        if self.fixed_slot(local, saved) {
            return Some(Source::Slot(local));
        }
        // Stop on depth due to cycles
        if depth > 8 {
            return None;
        }
        let Def::Once(location) = self.defs.defs[local] else { return None };
        match self.body.stmt_at(location) {
            // Skip temporaries that are only assigned once
            Either::Left(statement) => match &statement.kind {
                StatementKind::Assign(assign) => {
                    self.assigned_by_statement(&assign.1, saved, depth)
                }
                _ => None,
            },
            Either::Right(terminator) => match &terminator.kind {
                TerminatorKind::Call { func, args, .. } => {
                    self.assigned_by_call(func, args, saved, depth)
                }
                _ => None,
            },
        }
    }

    /// Where a reference assigned by `rvalue` leads
    fn assigned_by_statement(
        &self,
        rvalue: &Rvalue<'tcx>,
        saved: &DenseBitSet<Local>,
        depth: usize,
    ) -> Option<Source> {
        match rvalue {
            // A reborrow
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) if single_deref(*place) => {
                self.holder(place.local, saved, depth + 1)
            }
            // A borrow of a local
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) if !place.is_indirect() => {
                if !place.ty(self.body, self.tcx).ty.has_erased_regions() {
                    Some(Source::Nothing)
                } else if self.fixed_slot(place.local, saved) {
                    Some(Source::Slot(place.local))
                } else {
                    None
                }
            }
            // The same reference
            Rvalue::Use(operand, _) => self.operand_holder(operand, saved, depth + 1),
            _ => None,
        }
    }

    /// Where a reference returned by a call leads
    fn assigned_by_call(
        &self,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        saved: &DenseBitSet<Local>,
        depth: usize,
    ) -> Option<Source> {
        let (callee, _) = func.const_fn_def()?;
        if self.tcx.is_lang_item(callee, LangItem::GetContext) {
            // The executor's `Context`, which isn't the task's memory
            Some(Source::Nothing)
        } else if self.tcx.is_lang_item(callee, LangItem::PinNewUnchecked) {
            // `Pin::new_unchecked(r)` wraps the same reference
            self.operand_holder(&args.first()?.node, saved, depth + 1)
        } else {
            None
        }
    }

    /// A saved local that keeps its value until the checkpoint
    fn fixed_slot(&self, local: Local, saved: &DenseBitSet<Local>) -> bool {
        saved.contains(local) && self.defs.fixed.contains(local)
    }
}

/// Where a reference an instruction writes through leads, outside the future
#[derive(Clone, Copy)]
enum Source {
    /// Memory reached through the reference this saved local holds, which the checkpoint follows
    Slot(Local),
    /// Nothing outside the future: the future's own memory, or the executor's `Context`
    Nothing,
}

impl Source {
    fn slot(self) -> Option<Local> {
        match self {
            Source::Slot(local) => Some(local),
            Source::Nothing => None,
        }
    }
}

/// Whether `place` is stored in the future at a checkpoint whose saved locals are `saved`
fn in_future(place: Place<'_>, saved: &DenseBitSet<Local>) -> bool {
    place.local == CAPTURE_STRUCT_LOCAL || saved.contains(place.local)
}

/// Whether an instruction can write memory through a pointer
fn writes_through_pointers(effect: &Effect<'_>) -> bool {
    match *effect {
        Effect::Write(place, context) => place.is_indirect() || context == MutatingUseContext::Drop,
        Effect::Call | Effect::Intrinsic => true,
    }
}

/// Whether `place` is `(*_n).projections` with no other dereference. A second dereference reads a
/// pointer out of memory, which may have changed by the checkpoint.
fn single_deref(place: Place<'_>) -> bool {
    place.projection.first() == Some(&ProjectionElem::Deref)
        && place.projection.iter().filter(|elem| *elem == ProjectionElem::Deref).count() == 1
}

/// Where each local's own bytes are written, and which locals keep their value
struct LocalDefs {
    defs: IndexVec<Local, Def>,
    /// Locals are fixed if assigned once, outside any loop, and never borrowed in a way that could change them
    fixed: DenseBitSet<Local>,
}

#[derive(Clone, Copy)]
enum Def {
    Never,
    /// Assigned as a whole by exactly one statement or call
    Once(Location),
    Many,
}

impl LocalDefs {
    fn new<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut visitor = LocalDefsVisitor {
            tcx,
            body,
            defs: IndexVec::from_elem(Def::Never, &body.local_decls),
            borrowed: DenseBitSet::new_empty(body.local_decls.len()),
        };
        visitor.visit_body(body);
        let LocalDefsVisitor { defs, borrowed, .. } = visitor;

        let mut fixed = DenseBitSet::new_empty(body.local_decls.len());
        for (local, def) in defs.iter_enumerated() {
            if let Def::Once(location) = *def
                && !borrowed.contains(local)
                && !on_cycle(body, location.block)
            {
                fixed.insert(local);
            }
        }
        LocalDefs { defs, fixed }
    }
}

struct LocalDefsVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    defs: IndexVec<Local, Def>,
    /// Mutably borrowed, or shared-borrowed with interior mutability
    borrowed: DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for LocalDefsVisitor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Uses through a pointer reach the pointee, not the local
        if place.is_indirect() {
            return;
        }
        let local = place.local;
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Borrow | MutatingUseContext::RawBorrow,
            ) => {
                self.borrowed.insert(local);
            }
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::SharedBorrow | NonMutatingUseContext::RawBorrow,
            ) => {
                let (tcx, body) = (self.tcx, self.body);
                if !place.ty(body, tcx).ty.is_freeze(tcx, body.typing_env(tcx)) {
                    self.borrowed.insert(local);
                }
            }
            PlaceContext::MutatingUse(MutatingUseContext::Store | MutatingUseContext::Call)
                if place.projection.is_empty() =>
            {
                self.defs[local] = match self.defs[local] {
                    Def::Never => Def::Once(location),
                    Def::Once(_) | Def::Many => Def::Many,
                };
            }
            // Writes to part of the local, drops, resume values, ...
            PlaceContext::MutatingUse(_) => self.defs[local] = Def::Many,
            _ => {}
        }
    }
}

/// Whether `block` is part of a loop
fn on_cycle(body: &Body<'_>, block: BasicBlock) -> bool {
    let mut seen = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut stack: Vec<BasicBlock> = body[block].terminator().successors().collect();
    while let Some(bb) = stack.pop() {
        if bb == block {
            return true;
        }
        if seen.insert(bb) {
            stack.extend(body[bb].terminator().successors());
        }
    }
    false
}

fn print<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, plan: &CheckpointPlan<'tcx>) {
    eprintln!("=== rad write sets: {} ===", tcx.def_path_str(body.source.def_id()));
    let source_map = tcx.sess.source_map();
    if plan.loans.is_none() {
        eprintln!("  polonius: no live loans recorded (needs `-Zpolonius=next`)");
    }

    // Find instructions with duplicated borrow checker IDs or no ID
    let unnumbered = unnumbered(body);
    let (shared, duplicated) = repeated_ids(body);
    eprintln!(
        "  ids: {} without one {unnumbered:?}, {shared} shared by elaborated drops, {duplicated} duplicated",
        unnumbered.len()
    );

    let locals = |set: &DenseBitSet<Local>| -> String {
        set.iter().map(|local| format!("{local:?}")).collect::<Vec<_>>().join(", ")
    };
    for (index, awaited) in plan.awaits.iter().enumerate() {
        let suspend = awaited.suspend;
        let span =
            source_map.span_to_diagnostic_string(body[suspend].terminator().source_info.span);
        eprintln!("  .await {index} ({suspend:?}) at {span}");

        let starts = awaited
            .from_entry
            .then(|| "entry".to_string())
            .into_iter()
            .chain(awaited.after.iter().map(|earlier| format!("end of .await {earlier}")));
        eprintln!("    before, since {}:", starts.collect::<Vec<_>>().join(", "));
        print_write_set(tcx, body, plan.loans, &awaited.before);

        eprintln!("    effects, until {:?}:", awaited.ends);
        if !awaited.pending.is_empty() {
            eprintln!("      live, not in the future yet: {}", locals(&awaited.pending));
        }
        print_write_set(tcx, body, plan.loans, &awaited.effects);
    }
}

/// Prints the writes one checkpoint covers
fn print_write_set<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    loans: Option<&BodyLoans<'tcx>>,
    write_set: &WriteSet<'tcx>,
) {
    let saved: Vec<String> = write_set.saved.iter().map(|local| format!("{local:?}")).collect();
    eprintln!("      saved: {}", saved.join(", "));

    for write in &write_set.writes {
        let location = write.location;
        match write.effect {
            Effect::Write(place, context) => {
                let what = match context {
                    MutatingUseContext::Store => "store",
                    MutatingUseContext::SetDiscriminant => "discr",
                    MutatingUseContext::Call => "call dest",
                    MutatingUseContext::Yield => "resume",
                    MutatingUseContext::Drop => "drop",
                    MutatingUseContext::AsmOutput => "asm out",
                    _ => unreachable!("not recorded as a write: {context:?}"),
                };
                let how = access(tcx, body, place);
                let id = id_label(body, location);
                eprintln!("      {location:?} {id:<5} {what:<10} {how:<8} {place:?}");
            }
            Effect::Call => {
                let mut head = String::new();
                body[location.block].terminator().kind.fmt_head(&mut head).unwrap();
                let id = id_label(body, location);
                eprintln!("      {location:?} {id:<5} call       {head}");
            }
            Effect::Intrinsic => eprintln!("      {location:?} intrinsic  copy_nonoverlapping"),
        }
        if let (Some(reach), Some(loans)) = (&write.reach, loans) {
            print_live_loans(loans, reach, &write_set.saved);
        }
    }
    eprintln!("      ({} writes to locals not saved here)", write_set.not_saved);
}

/// Prints which loans the pointers used by an instruction can hold, and the memory those loans
/// resolve to
fn print_live_loans<'tcx>(
    loans: &BodyLoans<'tcx>,
    reach: &Reach<'tcx>,
    saved: &DenseBitSet<Local>,
) {
    match reach.instruction {
        Some(InstructionLoans { live, reachable: Some(reachable), outside }) => {
            let reachable: Vec<String> = reachable
                .iter()
                .filter_map(|loan| {
                    let (place, mutability) = loans.loans[loan]?;
                    Some(format!("{loan:?} &{}{place:?}", mutability.prefix_str()))
                })
                .collect();
            let outside = if *outside { " + outside" } else { "" };
            eprintln!(
                "                 reaches: {{{}}}{outside} (of {} live)",
                reachable.join(", "),
                live.count()
            );
        }
        _ => eprintln!("                 reaches: no borrowck result"),
    }
    print_targets(&reach.targets, saved);
}

/// Prints the memory an instruction can write through its pointers, split by whether it is
/// saved in the future at this checkpoint
fn print_targets(targets: &Targets<'_>, saved: &DenseBitSet<Local>) {
    let kept: Vec<String> = targets
        .places
        .iter()
        .filter(|&&place| in_future(place, saved))
        .map(|place| format!("{place:?}"))
        .collect();
    let unsaved = targets.places.len() - kept.len();
    let outside = if targets.outside { " + outside" } else { "" };
    let fallback = if targets.fallback { " (fallback)" } else { "" };
    eprintln!(
        "                 targets: {{{}}}{outside}{fallback}, dropped {} unsaved, {} frozen",
        kept.join(", "),
        unsaved,
        targets.frozen
    );
}

/// The memory a set of loans can point to
#[derive(Default)]
struct Targets<'tcx> {
    places: FxIndexSet<Place<'tcx>>,
    /// Memory this body didn't borrow
    outside: bool,
    /// The instruction had no borrowck result, so every loan still live was used instead
    fallback: bool,
    /// Skip shared loans of `Freeze` data
    frozen: usize,
}

/// Turns the loans an instruction's pointers can hold into the memory they point to
fn targets<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    loans: &BodyLoans<'tcx>,
    instruction: Option<&InstructionLoans>,
) -> Targets<'tcx> {
    let mut targets = Targets::default();
    match instruction {
        Some(InstructionLoans { reachable: Some(reachable), outside, .. }) => {
            targets.outside = *outside;
            for loan in reachable.iter() {
                targets.add(tcx, body, loans, loan);
            }
        }
        // Without a result, the pointers can hold any loan still live here.
        Some(InstructionLoans { live, .. }) => {
            targets.fallback = true;
            targets.outside = true;
            for loan in live.iter() {
                targets.add(tcx, body, loans, loan);
            }
        }
        // If no result was found above, every loan in the body is marked
        None => {
            targets.fallback = true;
            targets.outside = true;
            for loan in loans.loans.indices() {
                targets.add(tcx, body, loans, loan);
            }
        }
    }
    targets
}

impl<'tcx> Targets<'tcx> {
    fn add(&mut self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>, loans: &BodyLoans<'tcx>, loan: LoanId) {
        // Fake borrows point to nothing
        let Some((place, mutability)) = loans.loans[loan] else { return };
        if mutability == Mutability::Not
            && place.ty(body, tcx).ty.is_freeze(tcx, body.typing_env(tcx))
        {
            self.frozen += 1;
            return;
        }
        match place.iter_projections().rev().find(|(_, elem)| *elem == ProjectionElem::Deref) {
            None => {
                self.places.insert(place);
            }
            // Reborrow's loans are a superset of the original's
            Some((base, _)) if base.ty(body, tcx).ty.is_ref() => {}
            // Polonius can't resolve where a Box aliases
            Some(_) => self.outside = true,
        }
    }
}

/// The locals at the `index`th checkpoint
fn saved_at(body: &Body<'_>, liveness: &LivenessInfo, index: usize) -> DenseBitSet<Local> {
    let mut saved = DenseBitSet::new_empty(body.local_decls.len());
    for (saved_local, local) in liveness.saved_locals.iter_enumerated() {
        if liveness.live_locals_at_suspension_points[index].contains(saved_local) {
            saved.insert(local);
        }
    }
    saved
}

/// What an instruction in a segment does to memory
enum Effect<'tcx> {
    /// Writes the place, directly or through a pointer
    Write(Place<'tcx>, MutatingUseContext),
    /// A call, which also writes whatever the callee reaches through its arguments
    Call,
    /// `copy_nonoverlapping`, which writes through a pointer
    Intrinsic,
}

#[derive(Default)]
struct Effects<'tcx> {
    effects: Vec<(Location, Effect<'tcx>)>,
}

impl<'tcx> Visitor<'tcx> for Effects<'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        // Only match statements that change data
        if let PlaceContext::MutatingUse(
            context @ (MutatingUseContext::Store
            | MutatingUseContext::SetDiscriminant
            | MutatingUseContext::Call
            | MutatingUseContext::Yield
            | MutatingUseContext::Drop
            | MutatingUseContext::AsmOutput),
        ) = context
        {
            self.effects.push((location, Effect::Write(*place, context)));
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        if let StatementKind::Intrinsic(intrinsic) = &statement.kind
            && let NonDivergingIntrinsic::CopyNonOverlapping(..) = **intrinsic
        {
            self.effects.push((location, Effect::Intrinsic));
        }
        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call { .. } | TerminatorKind::TailCall { .. } = terminator.kind {
            self.effects.push((location, Effect::Call));
        }
        self.super_terminator(terminator, location);
    }
}

/// How an assignment reaches the memory it writes.
fn access<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, place: Place<'tcx>) -> &'static str {
    let Some((base, _)) = place.iter_projections().find(|(_, elem)| *elem == ProjectionElem::Deref)
    else {
        return "direct";
    };
    let ty = base.ty(body, tcx).ty;
    if ty.is_box() {
        "via Box"
    } else if ty.is_ref() {
        "via ref"
    } else {
        "via ptr"
    }
}
