use std::borrow::Cow;
use std::mem::discriminant;

use itertools::Itertools;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::Successors;
use rustc_data_structures::indexmap::{IndexMap, IndexSet};
use rustc_middle::mir::{
    AssertKind, BasicBlock, Body, Local, MirDumper, NonDivergingIntrinsic, OUTERMOST_SOURCE_SCOPE,
    Operand, Place, Rvalue, SourceInfo, Statement, StatementKind, TerminatorKind, WithRetag,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::Analysis;
use rustc_mir_dataflow::impls::{MaybeStorageLive, always_storage_live_locals};
use rustc_span::DUMMY_SP;
use tracing::instrument;

use crate::MirPass;
use crate::pass_manager::PassPolicy;
use crate::simplify::remove_dead_blocks;

pub(super) struct CollapseIdenticalYields;

impl<'tcx> MirPass<'tcx> for CollapseIdenticalYields {
    #[instrument(level = "debug", skip(self, tcx, body), ret)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.coroutine_kind().is_none() {
            return;
        }

        if let Some(dumper) = MirDumper::new(tcx, "collapse_yields_before", body) {
            dumper.dump_mir(body);
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

        let mut collapsed_yields = FxHashSet::default();

        for compare_yields in yields.iter().combinations(2) {
            let base_yield = compare_yields[0];
            let compare_yield = compare_yields[1];

            if collapsed_yields.contains(&compare_yield) {
                continue;
            }

            tracing::trace!(
                "Comparing yield {:?} to yield {:?}",
                base_yield.basic_block,
                compare_yield.basic_block
            );

            let Some(translation) = compare_yield.try_find_translation(&base_yield, body) else {
                // No translation, so these yields aren't equivalent
                continue;
            };
            let Some(translation) = translation.check_local_types(body) else {
                // The translated locals don't have the same types, so yields are not equivalent
                continue;
            };

            tracing::trace!(
                "Successfully translated yield {:?} to yield {:?}:\n{:?}",
                compare_yield.basic_block,
                base_yield.basic_block,
                translation
            );

            collapsed_yields.insert(compare_yield);

            translation.redirect_entry_points(tcx, body);
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
    fn try_find_translation(&self, other: &Yield, body: &Body<'_>) -> Option<TranslationMap> {
        let mut self_successors = self.all_successors(body);
        let mut other_successors = other.all_successors(body);

        let mut map = TranslationMap::new();

        for (self_successor, other_successor) in (&mut self_successors).zip(&mut other_successors) {
            if self_successor == other_successor {
                continue;
            }
            map = map.try_add_translation(self_successor, other_successor, body)?;
        }

        if self_successors.next().is_some() || other_successors.next().is_some() {
            // Can't be the same if they're not the same length
            return None;
        }

        Some(map)
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

    fn insert(mut self, l: Local, r: Local) -> Option<Self> {
        if let Some(old_r) = self.locals.insert(l, r) {
            if old_r != r {
                return None;
            }
        }
        Some(self)
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
        mut self,
        self_bb: BasicBlock,
        other_bb: BasicBlock,
        body: &Body<'_>,
    ) -> Option<Self> {
        let self_data = &body.basic_blocks[self_bb];
        let other_data = &body.basic_blocks[other_bb];

        if self_data.is_cleanup != other_data.is_cleanup {
            return None;
        }

        if self_data.statements.len() != other_data.statements.len() {
            return None;
        }

        if self_data.terminator.is_some() != other_data.terminator.is_some() {
            return None;
        }

        for (self_statement, other_statement) in
            self_data.statements.iter().zip(other_data.statements.iter())
        {
            self.locals = Self::statements_functionally_equivalent(
                self.locals,
                self_statement,
                other_statement,
            )?;
        }

        if let (Some(l_tk), Some(r_tk)) = (&self_data.terminator, &other_data.terminator) {
            self.locals = Self::terminator_kinds_functionally_equivalent(
                self.locals,
                &l_tk.kind,
                &r_tk.kind,
            )?;
        }

        self.blocks.insert(self_bb, other_bb);

        Some(self)
    }

    fn check_local_types(self, body: &Body<'_>) -> Option<Self> {
        for (l, r) in &self.locals.locals {
            if body.local_decls[*l].ty != body.local_decls[*r].ty {
                return None;
            }
        }

        Some(self)
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
                let predecessor_data = &mut body.basic_blocks_mut()[*predecessor];

                let live_locals = &from_live_locals[from];
                for from_local in live_locals.iter() {
                    if let Some(to_local) = self.locals.locals.get(&from_local) {
                        if from_local == *to_local {
                            continue;
                        }

                        if !always_live_locals.contains(*to_local) {
                            predecessor_data.statements.push(Statement::new(
                                SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                                StatementKind::StorageLive(*to_local),
                            ));
                        }
                        predecessor_data.statements.push(Statement::new(
                            SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                            StatementKind::Assign(Box::new((
                                Place::from(*to_local),
                                Rvalue::Use(Operand::Move(Place::from(from_local)), WithRetag::Yes),
                            ))),
                        ));
                        if !always_live_locals.contains(from_local) {
                            predecessor_data.statements.push(Statement::new(
                                SourceInfo { span: DUMMY_SP, scope: OUTERMOST_SOURCE_SCOPE },
                                StatementKind::StorageDead(from_local),
                            ));
                        }
                    }
                }

                // Change the terminator to go to the 'to' block
                predecessor_data.terminator_mut().successors_mut(|successor| {
                    if *successor == *from {
                        *successor = *to
                    }
                });
            }
        }
    }

    fn statements_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &Statement<'tcx>,
        r: &Statement<'tcx>,
    ) -> Option<LocalTranslationMap> {
        match (&l.kind, &r.kind) {
            (StatementKind::Assign(l_assign), StatementKind::Assign(r_assign)) => {
                map = map.insert(l_assign.0.local, r_assign.0.local)?;
                map = Self::rvalues_functionally_equivalent(map, &l_assign.1, &r_assign.1)?;
            }
            (StatementKind::FakeRead(l_fake_read), StatementKind::FakeRead(r_fake_read)) => {
                if l_fake_read.0 != r_fake_read.0 {
                    return None;
                }
                map = map.insert(l_fake_read.1.local, r_fake_read.1.local)?;
            }
            (
                StatementKind::SetDiscriminant { place: l_place, variant_index: l_variant_index },
                StatementKind::SetDiscriminant { place: r_place, variant_index: r_variant_index },
            ) => {
                if l_variant_index != r_variant_index {
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
            }
            (StatementKind::StorageLive(l_local), StatementKind::StorageLive(r_local)) => {
                map = map.insert(*l_local, *r_local)?;
            }
            (StatementKind::StorageDead(l_local), StatementKind::StorageDead(r_local)) => {
                map = map.insert(*l_local, *r_local)?;
            }
            (StatementKind::PlaceMention(l_place), StatementKind::PlaceMention(r_place)) => {
                map = map.insert(l_place.local, r_place.local)?;
            }
            (
                StatementKind::AscribeUserType(l_ascribe_user_type, l_variance),
                StatementKind::AscribeUserType(r_ascribe_user_type, r_variance),
            ) => {
                if l_ascribe_user_type.1 != r_ascribe_user_type.1 {
                    return None;
                }
                if l_variance != r_variance {
                    return None;
                }
                map = map.insert(l_ascribe_user_type.0.local, r_ascribe_user_type.0.local)?;
            }
            (
                StatementKind::Coverage(l_coverage_kind),
                StatementKind::Coverage(r_coverage_kind),
            ) => {
                if discriminant(l_coverage_kind) != discriminant(r_coverage_kind) {
                    return None;
                }
            }
            (
                StatementKind::Intrinsic(l_non_diverging_intrinsic),
                StatementKind::Intrinsic(r_non_diverging_intrinsic),
            ) => {
                map = Self::non_diverging_intrinsics_functionally_equivalent(
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
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
            }
            _ => {
                // By definition not equal
                return None;
            }
        }

        Some(map)
    }

    fn rvalues_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &Rvalue<'tcx>,
        r: &Rvalue<'tcx>,
    ) -> Option<LocalTranslationMap> {
        match (l, r) {
            (Rvalue::Use(l_operand, l_retag), Rvalue::Use(r_operand, r_retag)) => {
                if l_retag != r_retag {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (Rvalue::Repeat(l_operand, l_const), Rvalue::Repeat(r_operand, r_const)) => {
                if l_const != r_const {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (
                Rvalue::Ref(l_region, l_borrow_kind, l_place),
                Rvalue::Ref(r_region, r_borrow_kind, r_place),
            ) => {
                if l_region != r_region {
                    return None;
                }
                if l_borrow_kind != r_borrow_kind {
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
            }
            (Rvalue::ThreadLocalRef(l_def_id), Rvalue::ThreadLocalRef(r_def_id)) => {
                if l_def_id != r_def_id {
                    return None;
                }
            }
            (Rvalue::RawPtr(l_raw_ptr_kind, l_place), Rvalue::RawPtr(r_raw_ptr_kind, r_place)) => {
                if l_raw_ptr_kind != r_raw_ptr_kind {
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::Cast(l_cast_kind, l_operand, l_ty),
                Rvalue::Cast(r_cast_kind, r_operand, r_ty),
            ) => {
                if l_cast_kind != r_cast_kind {
                    return None;
                }
                if l_ty != r_ty {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (Rvalue::BinaryOp(l_bin_op, l_operands), Rvalue::BinaryOp(r_bin_op, r_operands)) => {
                if l_bin_op != r_bin_op {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, &l_operands.0, &r_operands.0)?;
                map = Self::operands_functionally_equivalent(map, &l_operands.1, &r_operands.1)?;
            }
            (Rvalue::UnaryOp(l_un_op, l_operand), Rvalue::UnaryOp(r_un_op, r_operand)) => {
                if l_un_op != r_un_op {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (Rvalue::Discriminant(l_place), Rvalue::Discriminant(r_place)) => {
                map = map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::Aggregate(l_aggregate_kind, l_index_vec),
                Rvalue::Aggregate(r_aggregate_kind, r_index_vec),
            ) => {
                if l_aggregate_kind != r_aggregate_kind {
                    return None;
                }
                if l_index_vec != r_index_vec {
                    return None;
                }
            }
            (Rvalue::CopyForDeref(l_place), Rvalue::CopyForDeref(r_place)) => {
                map = map.insert(l_place.local, r_place.local)?;
            }
            (
                Rvalue::WrapUnsafeBinder(l_operand, l_ty),
                Rvalue::WrapUnsafeBinder(r_operand, r_ty),
            ) => {
                if l_ty != r_ty {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (
                Rvalue::Reborrow(l_ty, l_mutability, l_place),
                Rvalue::Reborrow(r_ty, r_mutability, r_place),
            ) => {
                if l_ty != r_ty {
                    return None;
                }
                if l_mutability != r_mutability {
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
            }
            _ => {
                return None;
            }
        }

        Some(map)
    }

    fn operands_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &Operand<'tcx>,
        r: &Operand<'tcx>,
    ) -> Option<LocalTranslationMap> {
        match (l, r) {
            (Operand::Copy(l_place), Operand::Copy(r_place)) => {
                map = map.insert(l_place.local, r_place.local)?;
            }
            (Operand::Move(l_place), Operand::Move(r_place)) => {
                map = map.insert(l_place.local, r_place.local)?;
            }
            (Operand::Constant(l_const_operand), Operand::Constant(r_const_operand)) => {
                if l_const_operand.user_ty != r_const_operand.user_ty {
                    return None;
                }
                if l_const_operand.const_ != r_const_operand.const_ {
                    return None;
                }
            }
            (
                Operand::RuntimeChecks(l_runtime_checks),
                Operand::RuntimeChecks(r_runtime_checks),
            ) => {
                if l_runtime_checks != r_runtime_checks {
                    return None;
                }
            }
            _ => {
                return None;
            }
        }

        Some(map)
    }

    fn non_diverging_intrinsics_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &NonDivergingIntrinsic<'tcx>,
        r: &NonDivergingIntrinsic<'tcx>,
    ) -> Option<LocalTranslationMap> {
        match (l, r) {
            (
                NonDivergingIntrinsic::Assume(l_operand),
                NonDivergingIntrinsic::Assume(r_operand),
            ) => {
                map = Self::operands_functionally_equivalent(map, l_operand, r_operand)?;
            }
            (
                NonDivergingIntrinsic::CopyNonOverlapping(l_copy_non_overlapping),
                NonDivergingIntrinsic::CopyNonOverlapping(r_copy_non_overlapping),
            ) => {
                map = Self::operands_functionally_equivalent(
                    map,
                    &l_copy_non_overlapping.src,
                    &r_copy_non_overlapping.src,
                )?;
                map = Self::operands_functionally_equivalent(
                    map,
                    &l_copy_non_overlapping.dst,
                    &r_copy_non_overlapping.dst,
                )?;
                map = Self::operands_functionally_equivalent(
                    map,
                    &l_copy_non_overlapping.count,
                    &r_copy_non_overlapping.count,
                )?;
            }
            _ => {
                return None;
            }
        }

        Some(map)
    }

    fn terminator_kinds_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &TerminatorKind<'tcx>,
        r: &TerminatorKind<'tcx>,
    ) -> Option<LocalTranslationMap> {
        match (l, r) {
            (TerminatorKind::Goto { target: _ }, TerminatorKind::Goto { target: _ }) => {}
            (
                TerminatorKind::SwitchInt { discr: l_discr, targets: _ },
                TerminatorKind::SwitchInt { discr: r_discr, targets: _ },
            ) => {
                map = Self::operands_functionally_equivalent(map, l_discr, r_discr)?;
            }
            (TerminatorKind::UnwindResume, TerminatorKind::UnwindResume) => {}
            (
                TerminatorKind::UnwindTerminate(l_unwind_terminate_reason),
                TerminatorKind::UnwindTerminate(r_unwind_terminate_reason),
            ) => {
                if l_unwind_terminate_reason != r_unwind_terminate_reason {
                    return None;
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
                    return None;
                }
                if l_replace != r_replace {
                    return None;
                }
                map = map.insert(l_place.local, r_place.local)?;
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
                    return None;
                }
                if l_call_source != r_call_source {
                    return None;
                }
                if l_args.len() != r_args.len() {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_func, r_func)?;
                map = map.insert(l_destination.local, r_destination.local)?;
                for (l_arg, r_arg) in l_args.iter().zip(r_args.iter()) {
                    map = Self::operands_functionally_equivalent(map, &l_arg.node, &r_arg.node)?;
                }
            }
            (
                TerminatorKind::TailCall { func: l_func, args: l_args, fn_span: _ },
                TerminatorKind::TailCall { func: r_func, args: r_args, fn_span: _ },
            ) => {
                if l_args.len() != r_args.len() {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_func, r_func)?;
                for (l_arg, r_arg) in l_args.iter().zip(r_args.iter()) {
                    map = Self::operands_functionally_equivalent(map, &l_arg.node, &r_arg.node)?;
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
                    return None;
                }
                if discriminant(l_unwind) != discriminant(r_unwind) {
                    return None;
                }
                map = Self::operand_assert_kinds_functionally_equivalent(map, l_msg, r_msg)?;
                map = Self::operands_functionally_equivalent(map, l_cond, r_cond)?;
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
                map = Self::operands_functionally_equivalent(map, l_value, r_value)?;
                map = map.insert(l_resume_arg.local, r_resume_arg.local)?;
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
                    return None;
                }
            }
            (TerminatorKind::InlineAsm { .. }, TerminatorKind::InlineAsm { .. }) => {
                // Let's not risk messing with asm...
                return None;
            }
            _ => {
                return None;
            }
        }
        Some(map)
    }

    fn operand_assert_kinds_functionally_equivalent<'tcx>(
        mut map: LocalTranslationMap,
        l: &AssertKind<Operand<'tcx>>,
        r: &AssertKind<Operand<'tcx>>,
    ) -> Option<LocalTranslationMap> {
        match (l, r) {
            (
                AssertKind::BoundsCheck { len: l_len, index: l_index },
                AssertKind::BoundsCheck { len: r_len, index: r_index },
            ) => {
                map = Self::operands_functionally_equivalent(map, l_len, r_len)?;
                map = Self::operands_functionally_equivalent(map, l_index, r_index)?;
            }
            (
                AssertKind::Overflow(l_bin_op, l_op_0, l_op_1),
                AssertKind::Overflow(r_bin_op, r_op_0, r_op_1),
            ) => {
                if l_bin_op != r_bin_op {
                    return None;
                }
                map = Self::operands_functionally_equivalent(map, l_op_0, r_op_0)?;
                map = Self::operands_functionally_equivalent(map, l_op_1, r_op_1)?;
            }
            (AssertKind::OverflowNeg(l_op), AssertKind::OverflowNeg(r_op)) => {
                map = Self::operands_functionally_equivalent(map, l_op, r_op)?;
            }
            (AssertKind::DivisionByZero(l_op), AssertKind::DivisionByZero(r_op)) => {
                map = Self::operands_functionally_equivalent(map, l_op, r_op)?;
            }
            (AssertKind::RemainderByZero(l_op), AssertKind::RemainderByZero(r_op)) => {
                map = Self::operands_functionally_equivalent(map, l_op, r_op)?;
            }
            (
                AssertKind::ResumedAfterReturn(l_coroutine_kind),
                AssertKind::ResumedAfterReturn(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return None;
                }
            }
            (
                AssertKind::ResumedAfterPanic(l_coroutine_kind),
                AssertKind::ResumedAfterPanic(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return None;
                }
            }
            (
                AssertKind::ResumedAfterDrop(l_coroutine_kind),
                AssertKind::ResumedAfterDrop(r_coroutine_kind),
            ) => {
                if l_coroutine_kind != r_coroutine_kind {
                    return None;
                }
            }
            (
                AssertKind::MisalignedPointerDereference { required: l_required, found: l_found },
                AssertKind::MisalignedPointerDereference { required: r_required, found: r_found },
            ) => {
                map = Self::operands_functionally_equivalent(map, l_required, r_required)?;
                map = Self::operands_functionally_equivalent(map, l_found, r_found)?;
            }
            (AssertKind::NullPointerDereference, AssertKind::NullPointerDereference) => {}
            (AssertKind::NullReferenceConstructed, AssertKind::NullReferenceConstructed) => {}
            (
                AssertKind::InvalidEnumConstruction(l_op),
                AssertKind::InvalidEnumConstruction(r_op),
            ) => {
                map = Self::operands_functionally_equivalent(map, l_op, r_op)?;
            }
            _ => {
                return None;
            }
        }

        Some(map)
    }
}
