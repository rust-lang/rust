//! This pass statically detects code which has undefined behaviour or is likely to be erroneous.
//! It can be used to locate problems in MIR building or optimizations. It assumes that all code
//! can be executed, so it has false positives.
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::MaybeStorageLive;
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};
use std::borrow::Cow;

pub fn lint_body<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, when: String) {
    let reachable_blocks = traversal::reachable_as_bitset(body);
    let always_live_locals = &always_storage_live_locals(body);
    let storage_liveness = MaybeStorageLive::new(Cow::Borrowed(always_live_locals))
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    Lint {
        tcx,
        when,
        body,
        is_fn_like: tcx.def_kind(body.source.def_id()).is_fn_like(),
        always_live_locals,
        reachable_blocks,
        storage_liveness,
    }
    .visit_body(body);
}

struct Lint<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    when: String,
    body: &'a Body<'tcx>,
    is_fn_like: bool,
    always_live_locals: &'a BitSet<Local>,
    reachable_blocks: BitSet<BasicBlock>,
    storage_liveness: ResultsCursor<'a, 'tcx, MaybeStorageLive<'a>>,
}

impl<'a, 'tcx> Lint<'a, 'tcx> {
    #[track_caller]
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        let span = self.body.source_info(location).span;
        self.tcx.sess.dcx().span_delayed_bug(
            span,
            format!(
                "broken MIR in {:?} ({}) at {:?}:\n{}",
                self.body.source.instance,
                self.when,
                location,
                msg.as_ref()
            ),
        );
    }
}

impl<'a, 'tcx> Visitor<'tcx> for Lint<'a, 'tcx> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, location: Location) {
        if self.reachable_blocks.contains(location.block) && context.is_use() {
            self.storage_liveness.seek_after_primary_effect(location);
            if !self.storage_liveness.get().contains(local) {
                self.fail(location, format!("use of local {local:?}, which has no storage here"));
            }
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match statement.kind {
            StatementKind::StorageLive(local) => {
                if self.reachable_blocks.contains(location.block) {
                    self.storage_liveness.seek_before_primary_effect(location);
                    if self.storage_liveness.get().contains(local) {
                        self.fail(
                            location,
                            format!("StorageLive({local:?}) which already has storage here"),
                        );
                    }
                }
            }
            _ => {}
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match terminator.kind {
            TerminatorKind::Return => {
                if self.is_fn_like && self.reachable_blocks.contains(location.block) {
                    self.storage_liveness.seek_after_primary_effect(location);
                    for local in self.storage_liveness.get().iter() {
                        if !self.always_live_locals.contains(local) {
                            self.fail(
                                location,
                                format!(
                                    "local {local:?} still has storage when returning from function"
                                ),
                            );
                        }
                    }
                }
            }
            _ => {}
        }

        self.super_terminator(terminator, location);
    }
}
