//! Implementation of the [MergeYields] pass.
//!
//! The [super::coroutine::StateTransform] will take all yields in a body and make them into a
//! suspension point in the generated statemachine. This pass needs to run before that happens.
//!
//! The idea of this pass it to find all yield points and compare them.
//! If they are *functionally* identical, we can merge them. When that happens, the resulting
//! state machine will have fewer states, which is good for binary size and (probably)
//! performance.
//!
//! Take this code as example:
//! ```ignore (demonstration only)
//! if _0 {
//!     a(_1).await;
//! } else {
//!     a(_2).await;
//! }
//! ```
//!
//! This gets turned into this (simplified) MIR shape before the state transform:
//! ```txt
//!                      ┌─────────┐
//!             ┌────────┼switch _0┼────────┐
//!             │        └─────────┘        │
//!             │                           │
//!  ┌──────────▼──────────┐     ┌──────────▼───────────┐
//!  │create future A as _3│     │create future A as _4 │
//!  │with local _1        │     │with local _2         │
//!  └──────────┬──────────┘     └──────────┬───────────┘
//!         ┌───▼───┐                   ┌───▼───┐
//! ┌───────►poll _3│           ┌───────►poll _4│
//! │       └──┬─┬──┘           │       └──┬─┬──┘
//! │  ┌─────┐ │ │ ┌─────┐      │  ┌─────┐ │ │ ┌─────┐
//! └──┼yield◄─┘ └─►ready│      └──┼yield◄─┘ └─►ready│
//!    └─────┘     └──┬──┘         └─────┘     └──┬──┘
//!                   └───────┬───────────────────┘
//!                           │
//!                   ┌───────▼────────┐
//!                   │next thing to do│
//!                   └────────────────┘
//! ```
//!
//! To compare the yields, we take all the successors of each yield and walk them.
//! For each block we check if:
//! - The terminators are the same
//! - The statements are the same
//!
//! The only thing they're allowed to differ in are the indices of the locals and successor blocks.
//! Everything else must be the same and the locals must also consistently map onto each other.
//! So if we figured out during the walk that _40 maps to _50,
//! but then later we find out _40 now maps to _60, the walk is stopped and the yields are not
//! deemed identical.
//!
//! If this all went ok, we only need to check the translations of the locals and make sure
//! they're all of the same type.
//!
//! When we find identical yields, we need to merge them. This is done by taking all entry points
//! into the second yield and rewriting them to point into the first yield.
//!
//! Ultimately our example will look like this:
//! ```txt
//!                      ┌─────────┐
//!             ┌────────┼switch _0┼────────┐
//!             │        └─────────┘        │
//!             │                           │
//!  ┌──────────▼──────────┐     ┌──────────▼───────────┐
//!  │create future A as _3│     │create future A as _4 │
//!  │with local _1        │     │with local _2         │
//!  └──────────┬──────────┘     └──────────┬───────────┘
//!         ┌───▼───┐                 ┌─────▼──────┐
//! ┌───────►poll _3◄─────────────────┼_3 = move _4│
//! │       └──┬─┬──┘                 └────────────┘
//! │  ┌─────┐ │ │ ┌─────┐
//! └──┼yield◄─┘ └─►ready│
//!    └─────┘     └──┬──┘
//!                   └───────┐
//!                           │
//!                   ┌───────▼────────┐
//!                   │next thing to do│
//!                   └────────────────┘
//! ```
//!

use std::borrow::Cow;
use std::mem::discriminant;

use itertools::Itertools;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::Successors;
use rustc_data_structures::indexmap::{IndexMap, IndexSet};
use rustc_middle::mir::{
    AssertKind, BasicBlock, BasicBlockData, Body, Local, NonDivergingIntrinsic,
    OUTERMOST_SOURCE_SCOPE, Operand, Place, Rvalue, SourceInfo, Statement, StatementKind,
    Terminator, TerminatorKind, WithRetag,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::{MaybeStorageLive, always_storage_live_locals};
use rustc_span::DUMMY_SP;
use tracing::instrument;

use crate::MirPass;
use crate::pass_manager::PassPolicy;
use crate::simplify::remove_dead_blocks;

pub(super) struct MergeYields;

impl<'tcx> MirPass<'tcx> for MergeYields {
    #[instrument(level = "debug", skip(self, tcx, body), ret)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.coroutine_kind().is_none() {
            return;
        }

        tracing::debug!("running pass for {}", tcx.def_path_debug_str(body.source.def_id()));

        let mut yields = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(bb, bb_data)| {
                if let TerminatorKind::Yield { .. } = &bb_data.terminator.as_ref()?.kind {
                    Some(Yield { basic_block: bb })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        // Sort so we always translate from high bbs to low bbs
        yields.sort_unstable_by(|y1, y2| y1.basic_block.cmp(&y2.basic_block).reverse());

        let mut merged_yields = FxHashSet::default();

        for compare_yields in yields.iter().combinations(2) {
            let base_yield = compare_yields[0];
            let compare_yield = compare_yields[1];

            if merged_yields.contains(base_yield) || merged_yields.contains(compare_yield) {
                // Skip comparison if we've already merged this yield
                continue;
            }

            tracing::trace!(
                "Comparing yield {:?} to yield {:?}",
                base_yield.basic_block,
                compare_yield.basic_block
            );

            let Ok(translation) = compare_yield.try_find_translation(base_yield, body) else {
                // No translation, so these yields aren't equivalent
                continue;
            };
            if translation.check_local_types(body).is_err() {
                // The translated locals don't have the same types, so yields are not equivalent
                continue;
            };

            tracing::trace!(
                "Successfully translated yield {:?} to yield {:?}:\n{:?}",
                compare_yield.basic_block,
                base_yield.basic_block,
                translation
            );

            // The compare yield must be removed and every entry point is redirected to the equivalent entry into base
            // The removal itself is done at the end of the pass
            translation.redirect_entry_points(tcx, body);

            // Avoid comparing this yield again later since it has been removed
            merged_yields.insert(compare_yield);
        }

        remove_dead_blocks(body);
    }

    fn policy(&self, _sess: &rustc_session::Session) -> PassPolicy {
        PassPolicy::optimization(true)
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct Yield {
    basic_block: BasicBlock,
}

impl Yield {
    fn try_find_translation(
        &self,
        other: &Yield,
        body: &Body<'_>,
    ) -> Result<TranslationMap, TranslationError> {
        let mut self_successors = self.all_successors(body);
        let mut other_successors = other.all_successors(body);

        let mut map = TranslationMap::new();

        for (self_successor, other_successor) in (&mut self_successors).zip(&mut other_successors) {
            if self_successor == other_successor {
                continue;
            }
            map.try_add_translation(self_successor, other_successor, body)?;
        }

        if self_successors.next().is_some() || other_successors.next().is_some() {
            // Can't be the same if they're not the same length
            return Err(TranslationError);
        }

        Ok(map)
    }

    fn all_successors(&self, body: &Body<'_>) -> impl Iterator<Item = BasicBlock> {
        let mut seen: IndexSet<BasicBlock> = IndexSet::default();
        let mut todo = vec![self.basic_block];

        std::iter::from_fn(move || {
            let next = todo.pop()?;
            for successor in body.basic_blocks.successors(next) {
                if !seen.insert(successor) {
                    continue;
                }

                todo.push(successor);
            }
            Some(next)
        })
    }
}

#[derive(Debug)]
struct LocalTranslationMap {
    locals: IndexMap<Local, Local>,
}

impl LocalTranslationMap {
    fn new() -> Self {
        Self { locals: Default::default() }
    }

    /// Insert locals for translation.
    ///
    /// If `l` already exists but has a different `r` as value already,
    /// then None is returned. This signifies the translation has failed.
    fn insert(&mut self, l: Local, r: Local) -> Result<(), TranslationError> {
        if let Some(old_r) = self.locals.insert(l, r) {
            if old_r != r {
                return Err(TranslationError);
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct TranslationMap {
    blocks: IndexMap<BasicBlock, BasicBlock>,
    locals: LocalTranslationMap,
}

impl TranslationMap {
    fn new() -> Self {
        Self { blocks: Default::default(), locals: LocalTranslationMap::new() }
    }

    fn try_add_translation(
        &mut self,
        self_bb: BasicBlock,
        other_bb: BasicBlock,
        body: &Body<'_>,
    ) -> Result<(), TranslationError> {
        let self_data = &body.basic_blocks[self_bb];
        let other_data = &body.basic_blocks[other_bb];

        if self_data.is_cleanup != other_data.is_cleanup {
            return Err(TranslationError);
        }

        if self_data.statements.len() != other_data.statements.len() {
            return Err(TranslationError);
        }

        if self_data.terminator.is_some() != other_data.terminator.is_some() {
            return Err(TranslationError);
        }

        for (self_statement, other_statement) in
            self_data.statements.iter().zip(other_data.statements.iter())
        {
            Self::check_statements(&mut self.locals, self_statement, other_statement)?;
        }

        if let (Some(l_tk), Some(r_tk)) = (&self_data.terminator, &other_data.terminator) {
            Self::check_terminator_kinds(&mut self.locals, &l_tk.kind, &r_tk.kind)?;
        }

        self.blocks.insert(self_bb, other_bb);

        Ok(())
    }

    fn check_local_types(&self, body: &Body<'_>) -> Result<(), TranslationError> {
        for (l, r) in &self.locals.locals {
            if body.local_decls[*l].ty != body.local_decls[*r].ty {
                return Err(TranslationError);
            }
        }

        Ok(())
    }

    fn redirect_entry_points<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let body_predecessors = body.basic_blocks.predecessors().clone();
        let always_live_locals = always_storage_live_locals(body);
        let mut results = MaybeStorageLive::new(Cow::Borrowed(&always_live_locals))
            .iterate_to_fixpoint(tcx, body, Some("callapse_yields"))
            .into_results_cursor(body);

        let from_live_locals = self
            .blocks
            .keys()
            .map(|from| {
                results.seek_to_block_start(*from);
                (*from, results.get().clone())
            })
            .collect::<FxHashMap<_, _>>();

        for (from, to) in self.blocks.iter() {
            let predecessors = &body_predecessors[*from];
            // These predecessors go to the 'from' block, but need to go to the 'to' block
            // We also need to translate the live locals
            // So we insert the translations into the predecessor and change the successor to 'to'

            for predecessor in predecessors {
                let inbetween = body.basic_blocks_mut().push(BasicBlockData::new(
                    Some(Terminator {
                        source_info: SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                        kind: TerminatorKind::Goto { target: *to },
                        attributes: Default::default(),
                    }),
                    false,
                ));

                let predecessor_data = &mut body.basic_blocks_mut()[*predecessor];

                // Change the terminator to go to the 'to' block
                predecessor_data.terminator_mut().successors_mut(|successor| {
                    if *successor == *from {
                        *successor = inbetween;
                    }
                });
                let is_cleanup = body.basic_blocks_mut()[*from].is_cleanup;

                let inbetween_data = &mut body.basic_blocks_mut()[inbetween];
                inbetween_data.is_cleanup = is_cleanup;

                let live_locals = &from_live_locals[from];
                for from_local in live_locals.iter() {
                    if let Some(to_local) = self.locals.locals.get(&from_local) {
                        if from_local == *to_local {
                            continue;
                        }

                        if !always_live_locals.contains(*to_local) {
                            inbetween_data.statements.push(Statement::new(
                                SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                                StatementKind::StorageLive(*to_local),
                            ));
                        }
                        inbetween_data.statements.push(Statement::new(
                            SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                            StatementKind::Assign(Box::new((
                                Place::from(*to_local),
                                Rvalue::Use(Operand::Move(Place::from(from_local)), WithRetag::Yes),
                            ))),
                        ));
                        if !always_live_locals.contains(from_local) {
                            inbetween_data.statements.push(Statement::new(
                                SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                                StatementKind::StorageDead(from_local),
                            ));
                        }
                    }
                }
            }
        }
    }

    fn check_statements<'tcx>(
        map: &mut LocalTranslationMap,
        l: &Statement<'tcx>,
        r: &Statement<'tcx>,
    ) -> Result<(), TranslationError> {
        match (&l.kind, &r.kind) {
            (StatementKind::Assign(l_assign), StatementKind::Assign(r_assign)) => {
                map.insert(l_assign.0.local, r_assign.0.local)?;
                Self::check_rvalues(map, &l_assign.1, &r_assign.1)?;
            }
            (StatementKind::FakeRead(l_fake_read), StatementKind::FakeRead(r_fake_read)) => {
                if l_fake_read.0 != r_fake_read.0 {
                    return Err(TranslationError);
                }
                map.insert(l_fake_read.1.local, r_fake_read.1.local)?;
            }
            (
                StatementKind::SetDiscriminant { place: l_place, variant_index: l_variant_index },
                StatementKind::SetDiscriminant { place: r_place, variant_index: r_variant_index },
            ) => {
                if l_variant_index != r_variant_index {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            (StatementKind::StorageLive(l_local), StatementKind::StorageLive(r_local)) => {
                map.insert(*l_local, *r_local)?;
            }
            (StatementKind::StorageDead(l_local), StatementKind::StorageDead(r_local)) => {
                map.insert(*l_local, *r_local)?;
            }
            (StatementKind::PlaceMention(l_place), StatementKind::PlaceMention(r_place)) => {
                map.insert(l_place.local, r_place.local)?;
            }
            (
                StatementKind::AscribeUserType(l_ascribe_user_type, l_variance),
                StatementKind::AscribeUserType(r_ascribe_user_type, r_variance),
            ) => {
                if l_ascribe_user_type.1 != r_ascribe_user_type.1 {
                    return Err(TranslationError);
                }
                if l_variance != r_variance {
                    return Err(TranslationError);
                }
                map.insert(l_ascribe_user_type.0.local, r_ascribe_user_type.0.local)?;
            }
            (
                StatementKind::Coverage(l_coverage_kind),
                StatementKind::Coverage(r_coverage_kind),
            ) => {
                if discriminant(l_coverage_kind) != discriminant(r_coverage_kind) {
                    return Err(TranslationError);
                }
            }
            (
                StatementKind::Intrinsic(l_non_diverging_intrinsic),
                StatementKind::Intrinsic(r_non_diverging_intrinsic),
            ) => {
                Self::check_non_diverging_intrinsics(
                    map,
                    l_non_diverging_intrinsic,
                    r_non_diverging_intrinsic,
                )?;
            }
            (StatementKind::ConstEvalCounter, StatementKind::ConstEvalCounter) => {}
            (StatementKind::Nop, StatementKind::Nop) => {}
            (
                StatementKind::BackwardIncompatibleDropHint { place: l_place, reason: l_reason },
                StatementKind::BackwardIncompatibleDropHint { place: r_place, reason: r_reason },
            ) => {
                if l_reason != r_reason {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            _ => {
                // By definition not equal
                return Err(TranslationError);
            }
        }

        Ok(())
    }

    fn check_rvalues<'tcx>(
        map: &mut LocalTranslationMap,
        l: &Rvalue<'tcx>,
        r: &Rvalue<'tcx>,
    ) -> Result<(), TranslationError> {
        match (l, r) {
            (Rvalue::Use(l_operand, l_retag), Rvalue::Use(r_operand, r_retag)) => {
                if l_retag != r_retag {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (Rvalue::Repeat(l_operand, l_const), Rvalue::Repeat(r_operand, r_const)) => {
                if l_const != r_const {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (
                Rvalue::Ref(l_region, l_borrow_kind, l_place),
                Rvalue::Ref(r_region, r_borrow_kind, r_place),
            ) => {
                if l_region != r_region {
                    return Err(TranslationError);
                }
                if l_borrow_kind != r_borrow_kind {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            (Rvalue::ThreadLocalRef(l_def_id), Rvalue::ThreadLocalRef(r_def_id)) => {
                if l_def_id != r_def_id {
                    return Err(TranslationError);
                }
            }
            (Rvalue::RawPtr(l_raw_ptr_kind, l_place), Rvalue::RawPtr(r_raw_ptr_kind, r_place)) => {
                if l_raw_ptr_kind != r_raw_ptr_kind {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::Cast(l_cast_kind, l_operand, l_ty),
                Rvalue::Cast(r_cast_kind, r_operand, r_ty),
            ) => {
                if l_cast_kind != r_cast_kind {
                    return Err(TranslationError);
                }
                if l_ty != r_ty {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (Rvalue::BinaryOp(l_bin_op, l_operands), Rvalue::BinaryOp(r_bin_op, r_operands)) => {
                if l_bin_op != r_bin_op {
                    return Err(TranslationError);
                }
                Self::check_operands(map, &l_operands.0, &r_operands.0)?;
                Self::check_operands(map, &l_operands.1, &r_operands.1)?;
            }
            (Rvalue::UnaryOp(l_un_op, l_operand), Rvalue::UnaryOp(r_un_op, r_operand)) => {
                if l_un_op != r_un_op {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (Rvalue::Discriminant(l_place), Rvalue::Discriminant(r_place)) => {
                map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::Aggregate(l_aggregate_kind, l_index_vec),
                Rvalue::Aggregate(r_aggregate_kind, r_index_vec),
            ) => {
                if l_aggregate_kind != r_aggregate_kind {
                    return Err(TranslationError);
                }
                if l_index_vec != r_index_vec {
                    return Err(TranslationError);
                }
            }
            (Rvalue::CopyForDeref(l_place), Rvalue::CopyForDeref(r_place)) => {
                map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::WrapUnsafeBinder(l_operand, l_ty),
                Rvalue::WrapUnsafeBinder(r_operand, r_ty),
            ) => {
                if l_ty != r_ty {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (
                Rvalue::Reborrow(l_ty, l_mutability, l_place),
                Rvalue::Reborrow(r_ty, r_mutability, r_place),
            ) => {
                if l_ty != r_ty {
                    return Err(TranslationError);
                }
                if l_mutability != r_mutability {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            _ => {
                return Err(TranslationError);
            }
        }

        Ok(())
    }

    fn check_operands<'tcx>(
        map: &mut LocalTranslationMap,
        l: &Operand<'tcx>,
        r: &Operand<'tcx>,
    ) -> Result<(), TranslationError> {
        match (l, r) {
            (Operand::Copy(l_place), Operand::Copy(r_place)) => {
                map.insert(l_place.local, r_place.local)?;
            }
            (Operand::Move(l_place), Operand::Move(r_place)) => {
                map.insert(l_place.local, r_place.local)?;
            }
            (Operand::Constant(l_const_operand), Operand::Constant(r_const_operand)) => {
                if l_const_operand.user_ty != r_const_operand.user_ty {
                    return Err(TranslationError);
                }
                if l_const_operand.const_ != r_const_operand.const_ {
                    return Err(TranslationError);
                }
            }
            (
                Operand::RuntimeChecks(l_runtime_checks),
                Operand::RuntimeChecks(r_runtime_checks),
            ) => {
                if l_runtime_checks != r_runtime_checks {
                    return Err(TranslationError);
                }
            }
            _ => {
                return Err(TranslationError);
            }
        }

        Ok(())
    }

    fn check_non_diverging_intrinsics<'tcx>(
        map: &mut LocalTranslationMap,
        l: &NonDivergingIntrinsic<'tcx>,
        r: &NonDivergingIntrinsic<'tcx>,
    ) -> Result<(), TranslationError> {
        match (l, r) {
            (
                NonDivergingIntrinsic::Assume(l_operand),
                NonDivergingIntrinsic::Assume(r_operand),
            ) => {
                Self::check_operands(map, l_operand, r_operand)?;
            }
            (
                NonDivergingIntrinsic::CopyNonOverlapping(l_copy_non_overlapping),
                NonDivergingIntrinsic::CopyNonOverlapping(r_copy_non_overlapping),
            ) => {
                Self::check_operands(
                    map,
                    &l_copy_non_overlapping.src,
                    &r_copy_non_overlapping.src,
                )?;
                Self::check_operands(
                    map,
                    &l_copy_non_overlapping.dst,
                    &r_copy_non_overlapping.dst,
                )?;
                Self::check_operands(
                    map,
                    &l_copy_non_overlapping.count,
                    &r_copy_non_overlapping.count,
                )?;
            }
            _ => {
                return Err(TranslationError);
            }
        }

        Ok(())
    }

    fn check_terminator_kinds<'tcx>(
        map: &mut LocalTranslationMap,
        l: &TerminatorKind<'tcx>,
        r: &TerminatorKind<'tcx>,
    ) -> Result<(), TranslationError> {
        match (l, r) {
            (TerminatorKind::Goto { target: _ }, TerminatorKind::Goto { target: _ }) => {}
            (
                TerminatorKind::SwitchInt { discr: l_discr, targets: _ },
                TerminatorKind::SwitchInt { discr: r_discr, targets: _ },
            ) => {
                Self::check_operands(map, l_discr, r_discr)?;
            }
            (TerminatorKind::UnwindResume, TerminatorKind::UnwindResume) => {}
            (
                TerminatorKind::UnwindTerminate(l_unwind_terminate_reason),
                TerminatorKind::UnwindTerminate(r_unwind_terminate_reason),
            ) => {
                if l_unwind_terminate_reason != r_unwind_terminate_reason {
                    return Err(TranslationError);
                }
            }
            (TerminatorKind::Return, TerminatorKind::Return) => {}
            (TerminatorKind::Unreachable, TerminatorKind::Unreachable) => {}
            (
                TerminatorKind::Drop {
                    place: l_place,
                    target: _,
                    unwind: l_unwind,
                    replace: l_replace,
                    drop: _,
                },
                TerminatorKind::Drop {
                    place: r_place,
                    target: _,
                    unwind: r_unwind,
                    replace: r_replace,
                    drop: _,
                },
            ) => {
                if discriminant(l_unwind) != discriminant(r_unwind) {
                    return Err(TranslationError);
                }
                if l_replace != r_replace {
                    return Err(TranslationError);
                }
                map.insert(l_place.local, r_place.local)?;
            }
            (
                TerminatorKind::Call {
                    func: l_func,
                    args: l_args,
                    destination: l_destination,
                    target: _,
                    unwind: l_unwind,
                    call_source: l_call_source,
                    fn_span: _,
                },
                TerminatorKind::Call {
                    func: r_func,
                    args: r_args,
                    destination: r_destination,
                    target: _,
                    unwind: r_unwind,
                    call_source: r_call_source,
                    fn_span: _,
                },
            ) => {
                if discriminant(l_unwind) != discriminant(r_unwind) {
                    return Err(TranslationError);
                }
                if l_call_source != r_call_source {
                    return Err(TranslationError);
                }
                if l_args.len() != r_args.len() {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_func, r_func)?;
                map.insert(l_destination.local, r_destination.local)?;
                for (l_arg, r_arg) in l_args.iter().zip(r_args.iter()) {
                    Self::check_operands(map, &l_arg.node, &r_arg.node)?;
                }
            }
            (
                TerminatorKind::TailCall { func: l_func, args: l_args, fn_span: _ },
                TerminatorKind::TailCall { func: r_func, args: r_args, fn_span: _ },
            ) => {
                if l_args.len() != r_args.len() {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_func, r_func)?;
                for (l_arg, r_arg) in l_args.iter().zip(r_args.iter()) {
                    Self::check_operands(map, &l_arg.node, &r_arg.node)?;
                }
            }
            (
                TerminatorKind::Assert {
                    cond: l_cond,
                    expected: l_expected,
                    msg: l_msg,
                    target: _,
                    unwind: l_unwind,
                },
                TerminatorKind::Assert {
                    cond: r_cond,
                    expected: r_expected,
                    msg: r_msg,
                    target: _,
                    unwind: r_unwind,
                },
            ) => {
                if l_expected != r_expected {
                    return Err(TranslationError);
                }
                if discriminant(l_unwind) != discriminant(r_unwind) {
                    return Err(TranslationError);
                }
                Self::check_operand_assert_kinds(map, l_msg, r_msg)?;
                Self::check_operands(map, l_cond, r_cond)?;
            }
            (
                TerminatorKind::Yield {
                    value: l_value,
                    resume: _,
                    resume_arg: l_resume_arg,
                    drop: _,
                },
                TerminatorKind::Yield {
                    value: r_value,
                    resume: _,
                    resume_arg: r_resume_arg,
                    drop: _,
                },
            ) => {
                Self::check_operands(map, l_value, r_value)?;
                map.insert(l_resume_arg.local, r_resume_arg.local)?;
            }
            (TerminatorKind::CoroutineDrop, TerminatorKind::CoroutineDrop) => {}
            (
                TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ },
                TerminatorKind::FalseEdge { real_target: _, imaginary_target: _ },
            ) => {}
            (
                TerminatorKind::FalseUnwind { real_target: _, unwind: l_unwind },
                TerminatorKind::FalseUnwind { real_target: _, unwind: r_unwind },
            ) => {
                if discriminant(l_unwind) != discriminant(r_unwind) {
                    return Err(TranslationError);
                }
            }
            (TerminatorKind::InlineAsm { .. }, TerminatorKind::InlineAsm { .. }) => {
                // Let's not risk messing with asm...
                return Err(TranslationError);
            }
            _ => {
                return Err(TranslationError);
            }
        }
        Ok(())
    }

    fn check_operand_assert_kinds<'tcx>(
        map: &mut LocalTranslationMap,
        l: &AssertKind<Operand<'tcx>>,
        r: &AssertKind<Operand<'tcx>>,
    ) -> Result<(), TranslationError> {
        match (l, r) {
            (
                AssertKind::BoundsCheck { len: l_len, index: l_index },
                AssertKind::BoundsCheck { len: r_len, index: r_index },
            ) => {
                Self::check_operands(map, l_len, r_len)?;
                Self::check_operands(map, l_index, r_index)?;
            }
            (
                AssertKind::Overflow(l_bin_op, l_op_0, l_op_1),
                AssertKind::Overflow(r_bin_op, r_op_0, r_op_1),
            ) => {
                if l_bin_op != r_bin_op {
                    return Err(TranslationError);
                }
                Self::check_operands(map, l_op_0, r_op_0)?;
                Self::check_operands(map, l_op_1, r_op_1)?;
            }
            (AssertKind::OverflowNeg(l_op), AssertKind::OverflowNeg(r_op)) => {
                Self::check_operands(map, l_op, r_op)?;
            }
            (AssertKind::DivisionByZero(l_op), AssertKind::DivisionByZero(r_op)) => {
                Self::check_operands(map, l_op, r_op)?;
            }
            (AssertKind::RemainderByZero(l_op), AssertKind::RemainderByZero(r_op)) => {
                Self::check_operands(map, l_op, r_op)?;
            }
            (
                AssertKind::ResumedAfterReturn(l_coroutine_kind),
                AssertKind::ResumedAfterReturn(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return Err(TranslationError);
                }
            }
            (
                AssertKind::ResumedAfterPanic(l_coroutine_kind),
                AssertKind::ResumedAfterPanic(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return Err(TranslationError);
                }
            }
            (
                AssertKind::ResumedAfterDrop(l_coroutine_kind),
                AssertKind::ResumedAfterDrop(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return Err(TranslationError);
                }
            }
            (
                AssertKind::MisalignedPointerDereference { required: l_required, found: l_found },
                AssertKind::MisalignedPointerDereference { required: r_required, found: r_found },
            ) => {
                Self::check_operands(map, l_required, r_required)?;
                Self::check_operands(map, l_found, r_found)?;
            }
            (AssertKind::NullPointerDereference, AssertKind::NullPointerDereference) => {}
            (AssertKind::NullReferenceConstructed, AssertKind::NullReferenceConstructed) => {}
            (
                AssertKind::InvalidEnumConstruction(l_op),
                AssertKind::InvalidEnumConstruction(r_op),
            ) => {
                Self::check_operands(map, l_op, r_op)?;
            }
            _ => {
                return Err(TranslationError);
            }
        }

        Ok(())
    }
}

struct TranslationError;
