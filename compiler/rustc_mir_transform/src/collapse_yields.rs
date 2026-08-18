use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph::Successors;
use rustc_data_structures::indexmap::IndexSet;
use rustc_middle::mir::{
    BasicBlock, Body, Local, Operand, Place, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use tracing::instrument;

use crate::MirPass;
use crate::pass_manager::PassPolicy;

pub(super) struct CollapseIdenticalYields;

impl<'tcx> MirPass<'tcx> for CollapseIdenticalYields {
    #[instrument(level = "debug", skip(self, tcx, body), ret)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.coroutine_kind().is_none() {
            return;
        }

        tracing::debug!("running pass for {}", tcx.def_path_str(body.source.def_id()));

        let mut yields = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(bb, bb_data)| match &bb_data.terminator.as_ref()?.kind {
                TerminatorKind::Yield { value, resume, resume_arg, drop } => Some(Yield {
                    basic_block: bb,
                    value: value.clone(),
                    resume: *resume,
                    resume_arg: *resume_arg,
                    drop: *drop,
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        while let Some(base_yield) = yields.pop() {
            for i in (0..yields.len()).rev() {
                let compare_yield = &yields[i];

                let Some(translation) = base_yield.try_find_translation(compare_yield, body) else {
                    // No translation, so these yields aren't equivalent
                    continue;
                };

                // FIXME: impl doing the translation with a visitor
                tracing::trace!(
                    "Translate {:?} to {:?}: {:?}",
                    compare_yield.basic_block,
                    base_yield.basic_block,
                    translation
                );
            }
        }
    }

    fn policy(&self, _sess: &rustc_session::Session) -> PassPolicy {
        PassPolicy::optimization(true)
    }
}

#[derive(Debug)]
struct Yield<'tcx> {
    basic_block: BasicBlock,
    /// The value to return.
    value: Operand<'tcx>,
    /// Where to resume to.
    resume: BasicBlock,
    /// The place to store the resume argument in.
    resume_arg: Place<'tcx>,
    /// Cleanup to be done if the coroutine is dropped at this suspend point.
    drop: Option<BasicBlock>,
}

impl Yield<'_> {
    fn try_find_translation(&self, other: &Yield<'_>, body: &Body<'_>) -> Option<TranslationMap> {
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
struct TranslationMap {
    blocks: FxHashMap<BasicBlock, BasicBlock>,
    locals: FxHashMap<Local, Local>,
}

impl TranslationMap {
    fn new() -> Self {
        Self { blocks: Default::default(), locals: Default::default() }
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

        for (self_statement, other_statement) in
            self_data.statements.iter().zip(other_data.statements.iter())
        {
            let locals_map =
                Self::statements_functionally_equivalent(self_statement, other_statement)?;
            for (self_local, other_local) in locals_map {
                if let Some(old_other_local) = self.locals.insert(self_local, other_local) {
                    if old_other_local != other_local {
                        // We're not equivalent since a local has changed
                        return None;
                    }
                }
            }
        }

        // FIXME: Consider terminators

        self.blocks.insert(self_bb, other_bb);

        Some(self)
    }

    fn statements_functionally_equivalent(
        l: &Statement<'_>,
        r: &Statement<'_>,
    ) -> Option<FxHashMap<Local, Local>> {
        let mut map = FxHashMap::default();

        match (&l.kind, &r.kind) {
            (StatementKind::Assign(l_assign), StatementKind::Assign(r_assign)) => {
                map.insert(l_assign.0.local, r_assign.0.local);
                unimplemented!();
            }
            (StatementKind::FakeRead(l_fake_read), StatementKind::FakeRead(r_fake_read)) => {
                unimplemented!()
            }
            (
                StatementKind::SetDiscriminant { place: l_place, variant_index: l_variant_index },
                StatementKind::SetDiscriminant { place: r_place, variant_index: r_variant_index },
            ) => unimplemented!(),
            (StatementKind::StorageLive(l_local), StatementKind::StorageLive(r_local)) => {
                unimplemented!()
            }
            (StatementKind::StorageDead(l_local), StatementKind::StorageDead(r_local)) => {
                unimplemented!()
            }
            (StatementKind::PlaceMention(l_place), StatementKind::PlaceMention(r_place)) => {
                unimplemented!()
            }
            (
                StatementKind::AscribeUserType(l_ascribe_user_type, l_variance),
                StatementKind::AscribeUserType(r_ascribe_user_type, r_variance),
            ) => unimplemented!(),
            (
                StatementKind::Coverage(l_coverage_kind),
                StatementKind::Coverage(r_coverage_kind),
            ) => {
                unimplemented!()
            }
            (
                StatementKind::Intrinsic(l_non_diverging_intrinsic),
                StatementKind::Intrinsic(r_non_diverging_intrinsic),
            ) => unimplemented!(),
            (StatementKind::ConstEvalCounter, StatementKind::ConstEvalCounter) => {}
            (StatementKind::Nop, StatementKind::Nop) => {}
            (
                StatementKind::BackwardIncompatibleDropHint { place: l_place, reason: l_reason },
                StatementKind::BackwardIncompatibleDropHint { place: r_place, reason: r_reason },
            ) => unimplemented!(),
            _ => {}
        }

        Some(map)
    }
}
