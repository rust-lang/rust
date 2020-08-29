use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, BasicBlock, Location};
use rustc_middle::ty::TyCtxt;
use rustc_span::Span;

use super::ops;
use super::qualifs::{NeedsDrop, Qualif};
use super::validation::Qualifs;
use super::ConstCx;
use crate::dataflow::drop_flag_effects::on_all_drop_children_bits;
use crate::dataflow::move_paths::MoveData;
use crate::dataflow::{self, Analysis, MoveDataParamEnv, ResultsCursor};

/// Returns `true` if we should use the more precise live drop checker that runs after drop
/// elaboration.
pub fn checking_enabled(tcx: TyCtxt<'tcx>) -> bool {
    tcx.features().const_precise_live_drops
}

/// Look for live drops in a const context.
///
/// This is separate from the rest of the const checking logic because it must run after drop
/// elaboration.
pub fn check_live_drops(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &mir::Body<'tcx>) {
    let const_kind = tcx.hir().body_const_context(def_id);
    if const_kind.is_none() {
        return;
    }

    if !checking_enabled(tcx) {
        return;
    }

    let ccx = ConstCx { body, tcx, def_id, const_kind, param_env: tcx.param_env(def_id) };

    let mut visitor = CheckLiveDrops { ccx: &ccx, qualifs: Qualifs::default(), maybe_inits: None };

    visitor.visit_body(body);
}

type MaybeInits<'mir, 'tcx> = ResultsCursor<
    'mir,
    'tcx,
    dataflow::impls::MaybeInitializedPlaces<'mir, 'tcx, MoveDataParamEnv<'tcx>>,
>;

struct CheckLiveDrops<'mir, 'tcx> {
    ccx: &'mir ConstCx<'mir, 'tcx>,
    qualifs: Qualifs<'mir, 'tcx>,

    maybe_inits: Option<MaybeInits<'mir, 'tcx>>,
}

// So we can access `body` and `tcx`.
impl std::ops::Deref for CheckLiveDrops<'mir, 'tcx> {
    type Target = ConstCx<'mir, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.ccx
    }
}

impl CheckLiveDrops<'mir, 'tcx> {
    fn check_live_drop(&self, span: Span) {
        ops::non_const(self.ccx, ops::LiveDrop(None), span);
    }
}

impl Visitor<'tcx> for CheckLiveDrops<'mir, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, block: &mir::BasicBlockData<'tcx>) {
        trace!("visit_basic_block_data: bb={:?} is_cleanup={:?}", bb, block.is_cleanup);

        // Ignore drop terminators in cleanup blocks.
        if block.is_cleanup {
            return;
        }

        self.super_basic_block_data(bb, block);
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, location: Location) {
        trace!("visit_terminator: terminator={:?} location={:?}", terminator, location);

        match &terminator.kind {
            mir::TerminatorKind::Drop { place: dropped_place, .. } => {
                let dropped_ty = dropped_place.ty(self.body, self.tcx).ty;
                if !NeedsDrop::in_any_value_of_ty(self.ccx, dropped_ty) {
                    return;
                }

                if dropped_place.is_indirect() {
                    self.check_live_drop(terminator.source_info.span);
                    return;
                }

                if !self.qualifs.needs_drop(self.ccx, dropped_place.local, location) {
                    return;
                }

                let ConstCx { param_env, body, tcx, def_id, .. } = *self.ccx;

                // Replicate some logic from drop elaboration during const-checking. If we know
                // that the active variant of an enum does not have drop glue, we can allow it to
                // be dropped.
                let maybe_inits = self.maybe_inits.get_or_insert_with(|| {
                    let move_data = MoveData::gather_moves(body, tcx, param_env).unwrap();
                    let mdpe = MoveDataParamEnv { move_data, param_env };
                    dataflow::impls::MaybeInitializedPlaces::new(tcx, body, mdpe)
                        .mark_inactive_variants_as_uninit(true)
                        .into_engine(tcx, body, def_id.to_def_id())
                        .iterate_to_fixpoint()
                        .into_results_cursor(body)
                });
                maybe_inits.seek_before_primary_effect(location);
                let mdpe = &maybe_inits.analysis().mdpe;

                let dropped_mpi = mdpe
                    .move_data
                    .rev_lookup
                    .find(dropped_place.as_ref())
                    .expect_exact("All dropped places should have a move path");

                let mut is_live_drop = false;
                on_all_drop_children_bits(tcx, body, mdpe, dropped_mpi, |mpi| {
                    if maybe_inits.contains(mpi) {
                        is_live_drop = true;
                    }
                });

                if is_live_drop {
                    // Use the span where the dropped local was declared for the error.
                    let span = self.body.local_decls[dropped_place.local].source_info.span;
                    self.check_live_drop(span);
                }
            }

            mir::TerminatorKind::DropAndReplace { .. } => span_bug!(
                terminator.source_info.span,
                "`DropAndReplace` should be removed by drop elaboration",
            ),

            mir::TerminatorKind::Abort
            | mir::TerminatorKind::Call { .. }
            | mir::TerminatorKind::Assert { .. }
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. }
            | mir::TerminatorKind::GeneratorDrop
            | mir::TerminatorKind::Goto { .. }
            | mir::TerminatorKind::InlineAsm { .. }
            | mir::TerminatorKind::Resume
            | mir::TerminatorKind::Return
            | mir::TerminatorKind::SwitchInt { .. }
            | mir::TerminatorKind::Unreachable
            | mir::TerminatorKind::Yield { .. } => {}
        }
    }
}
