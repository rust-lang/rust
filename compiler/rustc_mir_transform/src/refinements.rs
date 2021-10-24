use std::ops::RangeInclusive;

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::{
    fmt::DebugWithContext, Analysis, AnalysisDomain, Engine, JoinSemiLattice, SwitchIntEdgeEffects,
};

pub struct Refinements;

impl<'tcx> MirPass<'tcx> for Refinements {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 4 {
            return;
        }

        let body_ref = &*body;
        let has_refine = HasRefinement { body: &body_ref };
        let mut results = Engine::new_generic(tcx, body_ref, has_refine)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        #[derive(Debug)]
        enum NewTerm {
            Unreachable,
            SwitchInt(Vec<(u128, BasicBlock)>, BasicBlock),
        }
        let mut mutate_buffer = Vec::new();
        for (bb, block) in body.basic_blocks().iter_enumerated() {
            let terminator_loc = body.terminator_loc(bb);
            if let TerminatorKind::SwitchInt { discr, targets, .. } = &block.terminator().kind {
                results.seek_before_primary_effect(terminator_loc);
                let state = results.get();

                let discr_refines = match discr.place() {
                    Some(discr_p) if discr_p.projection.is_empty() => {
                        state.0[discr_p.local.as_usize()]
                    }
                    _ => continue,
                };

                let mut new_targets = targets
                    .iter()
                    .filter(|&(value, _)| {
                        RefineRange::from(value..=value).is_subtype_of(discr_refines)
                    })
                    .collect::<Vec<_>>();

                let new_terminator = match RefineRange::for_otherwise_arm(discr_refines, &targets) {
                    None => match new_targets.pop() {
                        None => NewTerm::Unreachable,
                        Some((_, target_bb)) => NewTerm::SwitchInt(new_targets, target_bb),
                    },
                    Some(_) => NewTerm::SwitchInt(new_targets, targets.otherwise()),
                };

                match &new_terminator {
                    NewTerm::Unreachable => {
                        debug!("Refinements::run_pass: new_terminator=Unreachable for bb={:?}", bb)
                    }
                    NewTerm::SwitchInt(new_targets, _)
                        if new_targets.len() + 1 < targets.all_targets().len() =>
                    {
                        debug!(
                            "Refinements::run_pass: new_terminator={:?} for bb={:?}",
                            &new_terminator, bb,
                        );
                    }
                    _ => (),
                };

                mutate_buffer.push((bb, new_terminator));
            }
        }

        for (bb, new_terminator) in mutate_buffer.into_iter() {
            let block = &mut body.basic_blocks_mut()[bb];
            let term_kind = &mut block.terminator_mut().kind;
            let targets = term_kind.switch_targets_mut().unwrap();

            match new_terminator {
                NewTerm::Unreachable => *term_kind = TerminatorKind::Unreachable,
                NewTerm::SwitchInt(new_targets, otherwise) => {
                    *targets = SwitchTargets::new(new_targets.into_iter(), otherwise)
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// FIXME: handle signedness of refinements, currently we create *extremely*
/// large refinements for things like `-1..=0` which would give us `0..=uN::MAX`
struct RefineRange {
    start: u128,
    end: u128,
}

impl RefineRange {
    /// Returns a refinement that both `a` and `b` can be turned into
    /// i.e. for `mutual_supertype_of(0..=10, 5..=15)` we return `0..=15`
    /// (also known as union of the union of the two patterns)
    /// FIXME: for the case of `mutual_supertype_of(0..=10, 20..=30)`
    /// we return `0..=30`, eventually we should support "gapped"
    /// refinements and instead return `0..=10 | 20..=30`
    fn mutual_supertype_of(
        a: impl Into<Option<RefineRange>>,
        b: impl Into<Option<RefineRange>>,
    ) -> Option<Self> {
        let a = a.into();
        let b = b.into();
        match (a, b) {
            (None, other) | (other, None) => other,
            (Some(a), Some(b)) => {
                let range = (u128::min(a.start, b.start))..=(u128::max(a.end, b.end));
                Some(RefineRange::from(range))
            }
        }
    }

    fn is_subtype_of(self, sup: impl Into<Option<RefineRange>>) -> bool {
        match sup.into() {
            None => false,
            Some(sup) => sup.start <= self.start && sup.end >= self.end,
        }
    }

    /// Returns a refinement which can be turned into both `a` and `b`
    /// i.e. `mutual_subtype_of(0..=10, 5..=15)` would return `5..=10`
    /// (also known as the overlap/intersection of the two patterns)
    /// Returns `None` in the case where there is no overlap/subtype
    fn mutual_subtype_of(
        a: impl Into<Option<RefineRange>>,
        b: impl Into<Option<RefineRange>>,
    ) -> Option<Self> {
        let a = a.into()?;
        let b = b.into()?;
        Some(if a.is_subtype_of(b) {
            a
        } else if b.is_subtype_of(a) {
            b
        } else if a.start >= b.start && a.start <= b.end && a.end > b.end {
            // lhs point of `a` is inside of `b`
            RefineRange::from(a.start..=b.end)
        } else if b.start >= a.start && b.start <= a.end && b.end > a.end {
            // lhs point of `b` is inside of `a`
            RefineRange::from(b.start..=a.end)
        } else {
            // refine ranges are disjoint there is no mutual subtype
            // except for `None`
            return None;
        })
    }

    /// Returns a `RefineRange` representing the refinements that can be applied to the
    /// `switchInt`'d on place when the `otherwise` target is taken.
    fn for_otherwise_arm(
        discr_refines: Option<RefineRange>,
        targets: &SwitchTargets,
    ) -> Option<Self> {
        // FIXME this logic is kinda janky, i.e. if we have a range `0..=3`
        // with arms 2 and 3 in that order than we'll end up with a range `0..=2`
        // rather than `0..=1`... This is probably fine for now, this will all
        // be scrapped when we eventually support OR patterns properly.
        discr_refines.and_then(|mut discr_refines| {
            for (value, _) in targets.iter() {
                if value == discr_refines.start {
                    match discr_refines.start {
                        u128::MAX => return None,
                        _ => discr_refines.start += 1,
                    }
                }

                if value == discr_refines.end {
                    match discr_refines.end {
                        0 => return None,
                        _ => discr_refines.end -= 1,
                    }
                }

                if discr_refines.start > discr_refines.end {
                    return None;
                }
            }
            Some(discr_refines)
        })
    }
}

impl From<RangeInclusive<u128>> for RefineRange {
    fn from(range: RangeInclusive<u128>) -> Self {
        Self { start: *range.start(), end: *range.end() }
    }
}

struct HasRefinement<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> HasRefinement<'a, 'tcx> {
    fn get_switch_int_targets(&self, bb: BasicBlock) -> &SwitchTargets {
        match &self.body.basic_blocks()[bb].terminator().kind {
            TerminatorKind::SwitchInt { targets, .. } => targets,
            _ => panic!(""),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct LocalRefines(Box<[Option<RefineRange>]>);

impl LocalRefines {
    /// Sets every refinement to `..` / `_`
    fn clear_all(&mut self) {
        for refine in self.0.iter_mut() {
            *refine = Some(RefineRange::from(0..=(u128::MAX)));
        }
    }

    /// Sets refinement of `local` to `..` / `_`
    fn clear(&mut self, local: Local) {
        self.0[local.as_usize()] = Some(RefineRange::from(0..=(u128::MAX)));
    }

    fn set(&mut self, local: Local, refine: impl Into<Option<RefineRange>>) {
        self.0[local.as_usize()] = refine.into();
    }
}

impl JoinSemiLattice for LocalRefines {
    fn join(&mut self, other: &Self) -> bool {
        let mut mutated = false;
        for (lhs, rhs) in self.0.iter_mut().zip(other.0.iter()) {
            let sup = RefineRange::mutual_supertype_of(*lhs, *rhs);
            mutated |= *lhs != sup;
            *lhs = sup;
        }
        mutated
    }
}

impl<C> DebugWithContext<C> for LocalRefines {
    fn fmt_with(&self, _ctxt: &C, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl<'a, 'tcx> AnalysisDomain<'tcx> for HasRefinement<'a, 'tcx> {
    type Domain = LocalRefines;

    const NAME: &'static str = "has_refinement";

    fn bottom_value(&self, body: &rustc_middle::mir::Body<'tcx>) -> Self::Domain {
        // a refine of `None` conceptually represents a refinement like `is !` it gets
        // created whenever propagating refinements along edges that come from dead code:
        // the `2 => { ... }` arm below would get a refinement of `is !` on the local
        // `match 1 { 1 => { ...}, 2 => { ... }}`
        LocalRefines(vec![None; body.local_decls.len()].into_boxed_slice())
    }

    fn initialize_start_block(&self, _: &rustc_middle::mir::Body<'tcx>, state: &mut Self::Domain) {
        state.clear_all();
    }
}

impl<'a, 'tcx> Analysis<'tcx> for HasRefinement<'a, 'tcx> {
    fn apply_statement_effect(
        &self,
        trans: &mut Self::Domain,
        statement: &rustc_middle::mir::Statement<'tcx>,
        _: Location,
    ) {
        use StatementKind::*;
        match &statement.kind {
            SetDiscriminant { place, .. } => trans.clear(place.local),
            Assign(box (lhs_p, rhs)) => {
                trans.clear(lhs_p.local);

                use Rvalue::*;
                match rhs {
                    Use(Operand::Move(rhs_p) | Operand::Copy(rhs_p)) => {
                        if lhs_p.projection.is_empty() && rhs_p.projection.is_empty() {
                            let rhs_refines = trans.0[rhs_p.local.as_usize()];
                            trans.set(lhs_p.local, rhs_refines);
                        }
                    }
                    // FIXME this should probably do something
                    Use(Operand::Constant(_)) => (),
                    // clear the refinement regardless of mutability because `UnsafeCell` scares me
                    AddressOf(_, ref_p) => trans.clear(ref_p.local),
                    Ref(_, borrow_kind, ref_p) => {
                        match borrow_kind {
                            BorrowKind::Shared
                            // FIXME probably fine without projection refinements
                            | BorrowKind::Shallow
                            | BorrowKind::Unique => (),
                            BorrowKind::Mut { .. } => trans.clear(ref_p.local),
                        }
                    },
                    // FIXME this should probably do something
                    Aggregate(_, _) => (),
                    Repeat(_, _)
                    | ThreadLocalRef(_)
                    | Len(_)
                    | Discriminant(_)
                    | Cast(_, _, _)
                    | BinaryOp(_, _)
                    | CheckedBinaryOp(_, _)
                    | NullaryOp(_, _)
                    | UnaryOp(_, _)
                    | ShallowInitBox(_, _) => (),
                }
            }
            // FIXME can probably clear less than this
            LlvmInlineAsm(_) => trans.clear_all(),
            // I dont think this matters but can't hurt
            StorageDead(local) => trans.clear(*local),
            // doesn't matter
            FakeRead(_)
            | StorageLive(_)
            | Retag(_, _)
            | AscribeUserType(_, _)
            | Coverage(_)
            // CopyNonOverlapping isnt important because we don't have
            // refinements on borrows yet
            | CopyNonOverlapping(_)
            | Nop => (),
        }
    }

    fn apply_terminator_effect(
        &self,
        trans: &mut Self::Domain,
        terminator: &rustc_middle::mir::Terminator<'tcx>,
        _: Location,
    ) {
        use TerminatorKind::*;
        match &terminator.kind {
            Drop { place, .. } => trans.clear(place.local),
            // we can probably do better here and give `place` `value`'s refinements
            DropAndReplace { place, .. } => trans.clear(place.local),
            // this is probably overly conservative but sound
            InlineAsm { .. } => trans.clear_all(),
            // handled in `apply_call_return_effect`
            Call { .. } => (),
            // doesn't matter
            Goto { .. }
            | SwitchInt { .. }
            | Resume
            | Abort
            | Return
            | Unreachable
            | Assert { .. }
            | Yield { .. }
            | GeneratorDrop
            | FalseEdge { .. }
            | FalseUnwind { .. } => (),
        }
    }

    fn apply_call_return_effect(
        &self,
        trans: &mut Self::Domain,
        _: BasicBlock,
        _: &rustc_middle::mir::Operand<'tcx>,
        _: &[rustc_middle::mir::Operand<'tcx>],
        return_place: rustc_middle::mir::Place<'tcx>,
    ) {
        trans.clear(return_place.local);
    }

    #[instrument(level = "debug", skip(self, edge_effects))]
    fn apply_switch_int_edge_effects(
        &self,
        block: BasicBlock,
        discr: &rustc_middle::mir::Operand<'tcx>,
        edge_effects: &mut impl SwitchIntEdgeEffects<Self::Domain>,
    ) {
        edge_effects.apply(|trans, target| {
            if let Some(discr_p) = discr.place() {
                if discr_p.projection.is_empty() {
                    let discr_refine = trans.0[discr_p.local.as_usize()];
                    let refine = target
                        .value
                        .map(|v| Some(RefineRange::from(v..=v)))
                        .unwrap_or_else(|| {
                            RefineRange::for_otherwise_arm(
                                discr_refine,
                                self.get_switch_int_targets(block),
                            )
                        })
                        .and_then(|refine| RefineRange::mutual_subtype_of(refine, discr_refine));
                    debug!("applying refine={:?} on edge={:?}", refine, target,);
                    trans.set(discr_p.local, refine);
                }
            }
        });
    }
}
