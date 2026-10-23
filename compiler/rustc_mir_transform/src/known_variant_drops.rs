//! Experimental, local prototype: specializes `Drop` terminators of enum locals on the incoming
//! edges where the variant of the local is known.
//!
//! After `catch_unwind` is inlined, its `Result<R, Box<dyn Any + Send>>` is built in an `Ok` and
//! an `Err` block, joined, matched, and then dropped as a whole. The whole-value drop takes the
//! address of the local. That keeps the local out of `JumpThreading` (which ignores locals whose
//! address may be observed) and out of SROA in LLVM, so the drop glue of the `Err` payload sits on
//! the join path, looks hot, and gets inlined into every caller.
//!
//! On an edge where the variant is known, the drop is replaced by a drop of the only field of that
//! variant that needs dropping (moved to a fresh local first), or by a `goto` if no field needs
//! dropping. When every such drop is rewritten, the local is no longer address-taken and
//! `JumpThreading` can thread the match through the join, leaving the payload drop only on the
//! path that caught a panic.
//!
//! Only runs with `-Zmir-enable-passes=+KnownVariantDrops`.

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_const_eval::const_eval::DummyMachine;
use rustc_const_eval::interpret::InterpCx;
use rustc_data_structures::fx::FxIndexMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{DUMMY_SP, bug};
use tracing::{debug, trace};

use crate::PassPolicy;
use crate::simplify::simplify_cfg;

pub(super) struct KnownVariantDrops;

/// Drop blocks are duplicated once per known variant; larger blocks are left alone.
const MAX_DROP_BLOCK_STATEMENTS: usize = 32;

impl<'tcx> crate::MirPass<'tcx> for KnownVariantDrops {
    fn policy(&self, _ctx: &crate::PassCtx<'_>) -> PassPolicy {
        // Experimental: off unless requested with `-Zmir-enable-passes`.
        PassPolicy::optional(false)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        // Same restriction as `JumpThreading`, which this pass feeds.
        if tcx.is_coroutine(def_id) {
            return;
        }

        let typing_env = body.typing_env(tcx);
        let candidates = collect_candidates(tcx, typing_env, body);
        if candidates.is_empty() {
            return;
        }

        let rewrites = {
            let analysis = Analysis::new(typing_env, tcx, body, &candidates);
            let entry_states = analysis.entry_states();
            analysis.find_rewrites(&entry_states)
        };
        if rewrites.is_empty() {
            return;
        }

        debug!(?def_id, rewrites = rewrites.len(), "specializing drops of known variants");
        apply_rewrites(tcx, body, &candidates, &rewrites);
        simplify_cfg(tcx, body);
    }
}

/// What dropping the whole enum does when it holds a given variant.
#[derive(Copy, Clone, Debug)]
enum DropAction<'tcx> {
    /// No field of the variant needs dropping.
    Nothing,
    /// Exactly one field of the variant needs dropping.
    Field(FieldIdx, Ty<'tcx>),
    /// Several fields need dropping; the whole-value drop is kept.
    Whole,
}

struct Candidate<'tcx> {
    local: Local,
    adt: ty::AdtDef<'tcx>,
    actions: IndexVec<VariantIdx, DropAction<'tcx>>,
    /// Discriminant bit pattern of each variant, as compared by `SwitchInt` on
    /// `discriminant(local)`.
    discrs: Vec<(u128, VariantIdx)>,
}

/// Enum locals without a destructor that are dropped whole outside cleanup blocks and whose
/// address is not observed otherwise.
fn collect_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &Body<'tcx>,
) -> Vec<Candidate<'tcx>> {
    let mut dropped = DenseBitSet::new_empty(body.local_decls.len());
    for data in body.basic_blocks.iter() {
        if data.is_cleanup {
            continue;
        }
        if let TerminatorKind::Drop { place, drop: None, .. } = data.terminator().kind
            && let Some(local) = place.as_local()
        {
            dropped.insert(local);
        }
    }
    if dropped.is_empty() {
        return Vec::new();
    }

    let mut observed = AddressObserved { locals: DenseBitSet::new_empty(body.local_decls.len()) };
    observed.visit_body(body);

    let mut candidates = Vec::new();
    for local in dropped.iter() {
        if observed.locals.contains(local) {
            continue;
        }
        let ty::Adt(adt, args) = *body.local_decls[local].ty.kind() else { continue };
        if !adt.is_enum() || adt.has_dtor(tcx) {
            continue;
        }
        let actions: IndexVec<VariantIdx, DropAction<'tcx>> = adt
            .variants()
            .iter()
            .map(|variant| {
                let mut action = DropAction::Nothing;
                for (field_idx, field) in variant.fields.iter_enumerated() {
                    let field_ty = field.ty(tcx, args);
                    let field_ty = tcx
                        .try_normalize_erasing_regions(typing_env, field_ty)
                        .unwrap_or(field_ty.skip_norm_wip());
                    if !field_ty.needs_drop(tcx, typing_env) {
                        continue;
                    }
                    action = match action {
                        DropAction::Nothing => DropAction::Field(field_idx, field_ty),
                        DropAction::Field(..) | DropAction::Whole => DropAction::Whole,
                    };
                }
                action
            })
            .collect();
        if actions.iter().all(|action| matches!(action, DropAction::Whole)) {
            continue;
        }
        let discrs = adt.discriminants(tcx).map(|(variant, discr)| (discr.val, variant)).collect();
        trace!(?local, ?actions, "candidate");
        candidates.push(Candidate { local, adt, actions, discrs });
    }
    candidates
}

/// Locals whose address may be observed by anything other than a drop of the whole local.
struct AddressObserved {
    locals: DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for AddressObserved {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        let whole_drop = context == PlaceContext::MutatingUse(MutatingUseContext::Drop)
            && place.projection.is_empty();
        if context.may_observe_address() && !place.is_indirect() && !whole_drop {
            self.locals.insert(place.local);
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Val {
    /// No execution reaches this point.
    Unreached,
    Known(VariantIdx),
    Unknown,
}

impl Val {
    fn join(self, other: Val) -> Val {
        match (self, other) {
            (Val::Unreached, v) | (v, Val::Unreached) => v,
            (Val::Known(a), Val::Known(b)) if a == b => self,
            _ => Val::Unknown,
        }
    }

    /// Narrows `self` with what taking a switch edge implies.
    fn meet(self, other: Val) -> Val {
        match (self, other) {
            (Val::Unknown, v) | (v, Val::Unknown) => v,
            (Val::Known(a), Val::Known(b)) if a == b => self,
            _ => Val::Unreached,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Edge {
    from: BasicBlock,
    /// Index into `SwitchTargets::all_targets` when `from` ends in a `SwitchInt`.
    slot: Option<usize>,
    target: BasicBlock,
}

struct Rewrite {
    drop_block: BasicBlock,
    candidate: usize,
    variant: VariantIdx,
    edges: Vec<Edge>,
}

/// Forgets the variant of every candidate that the visited code may write.
struct Clobber<'a> {
    index: &'a IndexSlice<Local, Option<usize>>,
    state: &'a mut [Val],
}

impl<'tcx> Visitor<'tcx> for Clobber<'_> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        if context.is_mutating_use()
            && let Some(i) = self.index[place.local]
        {
            self.state[i] = Val::Unknown;
        }
    }

    fn visit_local(&mut self, local: Local, context: PlaceContext, _location: Location) {
        if (context.is_mutating_use() || context.is_storage_marker())
            && let Some(i) = self.index[local]
        {
            self.state[i] = Val::Unknown;
        }
    }
}

/// Whether `statement` may change the value or the storage of `local`.
fn writes_local<'tcx>(statement: &Statement<'tcx>, local: Local) -> bool {
    struct Finder {
        local: Local,
        found: bool,
    }

    impl<'tcx> Visitor<'tcx> for Finder {
        fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
            if place.local == self.local && context.is_mutating_use() {
                self.found = true;
            }
        }

        fn visit_local(&mut self, local: Local, context: PlaceContext, _location: Location) {
            if local == self.local && (context.is_mutating_use() || context.is_storage_marker()) {
                self.found = true;
            }
        }
    }

    let mut finder = Finder { local, found: false };
    finder.visit_statement(statement, START_BLOCK.start_location());
    finder.found
}

struct Analysis<'a, 'tcx> {
    body: &'a Body<'tcx>,
    ecx: InterpCx<'tcx, DummyMachine>,
    candidates: &'a [Candidate<'tcx>],
    /// Candidate index of each local.
    index: IndexVec<Local, Option<usize>>,
}

impl<'a, 'tcx> Analysis<'a, 'tcx> {
    fn new(
        typing_env: ty::TypingEnv<'tcx>,
        tcx: TyCtxt<'tcx>,
        body: &'a Body<'tcx>,
        candidates: &'a [Candidate<'tcx>],
    ) -> Self {
        let mut index = IndexVec::from_elem(None, &body.local_decls);
        for (i, candidate) in candidates.iter().enumerate() {
            index[candidate.local] = Some(i);
        }
        Analysis {
            body,
            ecx: InterpCx::new(tcx, DUMMY_SP, typing_env, DummyMachine),
            candidates,
            index,
        }
    }

    fn assigned_variant(
        &self,
        candidate: &Candidate<'tcx>,
        rvalue: &Rvalue<'tcx>,
    ) -> Option<VariantIdx> {
        match rvalue {
            Rvalue::Aggregate(kind, _) => match **kind {
                AggregateKind::Adt(did, variant, _, _, None) if did == candidate.adt.did() => {
                    Some(variant)
                }
                _ => None,
            },
            // `Ok(())` and other field-less values are constants after GVN.
            Rvalue::Use(Operand::Constant(constant), _) => {
                let op = self
                    .ecx
                    .eval_mir_constant(&constant.const_, constant.span, None)
                    .discard_err()?;
                self.ecx.read_discriminant(&op).discard_err()
            }
            _ => None,
        }
    }

    fn apply_statement(&self, statement: &Statement<'tcx>, location: Location, state: &mut [Val]) {
        Clobber { index: &self.index, state: &mut *state }.visit_statement(statement, location);
        match &statement.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = &**assign;
                if let Some(local) = place.as_local()
                    && let Some(i) = self.index[local]
                {
                    state[i] = self
                        .assigned_variant(&self.candidates[i], rvalue)
                        .map_or(Val::Unknown, Val::Known);
                }
            }
            StatementKind::SetDiscriminant { place, variant_index } => {
                if let Some(local) = place.as_local()
                    && let Some(i) = self.index[local]
                {
                    state[i] = Val::Known(*variant_index);
                }
            }
            _ => {}
        }
    }

    fn apply_statements(&self, bb: BasicBlock, state: &mut [Val]) {
        for (statement_index, statement) in self.body.basic_blocks[bb].statements.iter().enumerate()
        {
            self.apply_statement(statement, Location { block: bb, statement_index }, state);
        }
    }

    /// For `switchInt(_d)` where the last write of `_d` in `bb` is `_d = discriminant(P)` and
    /// neither `_d` nor `P` is written after it, returns P's candidate index and the value of P
    /// implied by each slot of `targets.all_targets()`.
    fn switch_refinement(
        &self,
        bb: BasicBlock,
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
    ) -> Option<(usize, Vec<Val>)> {
        let discr_local = discr.place()?.as_local()?;
        let statements = &self.body.basic_blocks[bb].statements;
        let (position, i) =
            statements.iter().enumerate().rev().find_map(|(position, statement)| {
                if !writes_local(statement, discr_local) {
                    return None;
                }
                let found = if let StatementKind::Assign(assign) = &statement.kind
                    && let (place, Rvalue::Discriminant(enum_place)) = &**assign
                    && place.as_local() == Some(discr_local)
                    && let Some(enum_local) = enum_place.as_local()
                    && let Some(i) = self.index[enum_local]
                {
                    Some((position, i))
                } else {
                    None
                };
                Some(found)
            })??;
        let candidate = &self.candidates[i];
        if statements[position + 1..]
            .iter()
            .any(|s| writes_local(s, discr_local) || writes_local(s, candidate.local))
        {
            return None;
        }

        let mut remaining: Vec<VariantIdx> = candidate.discrs.iter().map(|&(_, v)| v).collect();
        let mut values = Vec::with_capacity(targets.all_targets().len());
        for (value, _) in targets.iter() {
            // Give up on values that match no variant instead of guessing what they mean.
            let &(_, variant) = candidate.discrs.iter().find(|&&(d, _)| d == value)?;
            values.push(Val::Known(variant));
            remaining.retain(|&v| v != variant);
        }
        values.push(match remaining[..] {
            [] => Val::Unreached,
            [variant] => Val::Known(variant),
            _ => Val::Unknown,
        });
        Some((i, values))
    }

    /// Calls `f` with the state on every edge from `bb` into a non-cleanup block, given the
    /// state before `bb`'s terminator.
    fn for_each_edge(&self, bb: BasicBlock, state: &[Val], mut f: impl FnMut(Edge, &[Val])) {
        let data = &self.body.basic_blocks[bb];
        let terminator = data.terminator();
        let location = Location { block: bb, statement_index: data.statements.len() };
        let mut exit = state.to_vec();
        Clobber { index: &self.index, state: &mut exit }.visit_terminator(terminator, location);

        if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind {
            let refinement = self.switch_refinement(bb, discr, targets);
            let mut edge_state = exit.clone();
            for (slot, &target) in targets.all_targets().iter().enumerate() {
                if self.body.basic_blocks[target].is_cleanup {
                    continue;
                }
                edge_state.copy_from_slice(&exit);
                if let Some((i, values)) = &refinement {
                    edge_state[*i] = exit[*i].meet(values[slot]);
                }
                f(Edge { from: bb, slot: Some(slot), target }, &edge_state);
            }
        } else {
            for target in terminator.successors() {
                if self.body.basic_blocks[target].is_cleanup {
                    continue;
                }
                f(Edge { from: bb, slot: None, target }, &exit);
            }
        }
    }

    /// Forward dataflow to a fixpoint. `None` means that no edge reaches the block.
    fn entry_states(&self) -> IndexVec<BasicBlock, Option<Vec<Val>>> {
        let mut entry: IndexVec<BasicBlock, Option<Vec<Val>>> =
            IndexVec::from_elem(None, &self.body.basic_blocks);
        entry[START_BLOCK] = Some(vec![Val::Unknown; self.candidates.len()]);
        let rpo = self.body.basic_blocks.reverse_postorder();
        let mut changed = true;
        while changed {
            changed = false;
            for &bb in rpo {
                if self.body.basic_blocks[bb].is_cleanup {
                    continue;
                }
                let Some(mut state) = entry[bb].clone() else { continue };
                self.apply_statements(bb, &mut state);
                self.for_each_edge(bb, &state, |edge, edge_state| {
                    let target_state = &mut entry[edge.target];
                    match target_state {
                        Some(old) => {
                            for (old, &new) in old.iter_mut().zip(edge_state) {
                                let joined = old.join(new);
                                changed |= joined != *old;
                                *old = joined;
                            }
                        }
                        None => {
                            *target_state = Some(edge_state.to_vec());
                            changed = true;
                        }
                    }
                });
            }
        }
        entry
    }

    fn find_rewrites(&self, entry: &IndexSlice<BasicBlock, Option<Vec<Val>>>) -> Vec<Rewrite> {
        // Drop block -> (candidate index, incoming edges with their states).
        let mut drop_blocks: FxIndexMap<BasicBlock, (usize, Vec<(Edge, Vec<Val>)>)> =
            FxIndexMap::default();
        for (bb, data) in self.body.basic_blocks.iter_enumerated() {
            if data.is_cleanup || data.statements.len() > MAX_DROP_BLOCK_STATEMENTS {
                continue;
            }
            if let TerminatorKind::Drop { place, replace: false, drop: None, .. } =
                data.terminator().kind
                && let Some(local) = place.as_local()
                && let Some(i) = self.index[local]
            {
                drop_blocks.insert(bb, (i, Vec::new()));
            }
        }
        if drop_blocks.is_empty() {
            return Vec::new();
        }

        for &bb in self.body.basic_blocks.reverse_postorder() {
            if self.body.basic_blocks[bb].is_cleanup {
                continue;
            }
            let Some(state) = &entry[bb] else { continue };
            let mut state = state.clone();
            self.apply_statements(bb, &mut state);
            self.for_each_edge(bb, &state, |edge, edge_state| {
                if let Some((_, incoming)) = drop_blocks.get_mut(&edge.target) {
                    incoming.push((edge, edge_state.to_vec()));
                }
            });
        }

        let mut rewrites = Vec::new();
        for (drop_block, (i, incoming)) in drop_blocks {
            let candidate = &self.candidates[i];
            let mut groups: FxIndexMap<VariantIdx, Vec<Edge>> = FxIndexMap::default();
            for (edge, mut state) in incoming {
                if edge.from == drop_block {
                    continue;
                }
                self.apply_statements(drop_block, &mut state);
                if let Val::Known(variant) = state[i]
                    && !matches!(candidate.actions[variant], DropAction::Whole)
                {
                    groups.entry(variant).or_default().push(edge);
                }
            }
            for (variant, edges) in groups {
                trace!(?drop_block, ?variant, ?edges, "rewrite");
                rewrites.push(Rewrite { drop_block, candidate: i, variant, edges });
            }
        }
        rewrites
    }
}

fn apply_rewrites<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    candidates: &[Candidate<'tcx>],
    rewrites: &[Rewrite],
) {
    // Clone every drop block before retargeting any edge, so that each copy keeps the original
    // successors.
    let mut new_blocks = Vec::with_capacity(rewrites.len());
    for rewrite in rewrites {
        let candidate = &candidates[rewrite.candidate];
        let mut data = body.basic_blocks[rewrite.drop_block].clone();
        let source_info = data.terminator().source_info;
        let TerminatorKind::Drop { place, target, unwind, replace, drop: None } =
            data.terminator().kind
        else {
            bug!("expected a drop terminator in {:?}", rewrite.drop_block);
        };
        let kind = match candidate.actions[rewrite.variant] {
            DropAction::Nothing => {
                TerminatorKind::Goto { target, loop_hint_attrs: Default::default() }
            }
            DropAction::Field(field, field_ty) => {
                // Dropping a moved-out copy is fine: the local's address is never observed, so
                // nothing can rely on the field being dropped in place.
                let temp = body.local_decls.push(LocalDecl::new(field_ty, source_info.span));
                let field_place = tcx.mk_place_field(
                    tcx.mk_place_downcast(place, candidate.adt, rewrite.variant),
                    field,
                    field_ty,
                );
                let mut statement = Statement::new(
                    source_info,
                    StatementKind::Assign(Box::new((
                        Place::from(temp),
                        Rvalue::Use(Operand::Move(field_place), WithRetag::Yes),
                    ))),
                );
                statement.debuginfos.prepend(&mut data.after_last_stmt_debuginfos);
                data.statements.push(statement);
                TerminatorKind::Drop {
                    place: Place::from(temp),
                    target,
                    unwind,
                    replace,
                    drop: None,
                }
            }
            DropAction::Whole => bug!("whole-value drops are not rewritten"),
        };
        data.terminator_mut().kind = kind;
        new_blocks.push(body.basic_blocks.as_mut().push(data));
    }

    for (rewrite, &new_block) in rewrites.iter().zip(&new_blocks) {
        for edge in &rewrite.edges {
            let terminator = body.basic_blocks.as_mut()[edge.from].terminator_mut();
            if let Some(slot) = edge.slot {
                let TerminatorKind::SwitchInt { targets, .. } = &mut terminator.kind else {
                    bug!("switch edge from a non-switch terminator in {:?}", edge.from);
                };
                let target = &mut targets.all_targets_mut()[slot];
                assert_eq!(*target, rewrite.drop_block);
                *target = new_block;
            } else {
                let drop_block = rewrite.drop_block;
                terminator.successors_mut(|successor| {
                    if *successor == drop_block {
                        *successor = new_block;
                    }
                });
            }
        }
    }
}
