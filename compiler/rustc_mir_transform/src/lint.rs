//! This pass statically detects code which has undefined behaviour or is likely to be erroneous.
//! It can be used to locate problems in MIR building or optimizations. It assumes that all code
//! can be executed, so it has false positives.

use std::borrow::Cow;

use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::impls::{MaybeStorageDead, MaybeStorageLive};
use rustc_mir_dataflow::storage::always_storage_live_locals;
use rustc_mir_dataflow::{Analysis, ResultsCursor};

pub(super) fn lint_body<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, when: String) {
    let always_live_locals = &always_storage_live_locals(body);

    let maybe_storage_live = MaybeStorageLive::new(Cow::Borrowed(always_live_locals))
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    let maybe_storage_dead = MaybeStorageDead::new(Cow::Borrowed(always_live_locals))
        .into_engine(tcx, body)
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    let mut lint = Lint {
        tcx,
        when,
        body,
        is_fn_like: tcx.def_kind(body.source.def_id()).is_fn_like(),
        always_live_locals,
        maybe_storage_live,
        maybe_storage_dead,
        places: Default::default(),
    };
    for (bb, data) in traversal::reachable(body) {
        lint.visit_basic_block_data(bb, data);
    }
}

struct Lint<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    when: String,
    body: &'a Body<'tcx>,
    is_fn_like: bool,
    always_live_locals: &'a BitSet<Local>,
    maybe_storage_live: ResultsCursor<'a, 'tcx, MaybeStorageLive<'a>>,
    maybe_storage_dead: ResultsCursor<'a, 'tcx, MaybeStorageDead<'a>>,
    places: FxHashSet<PlaceRef<'tcx>>,
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
        if context.is_use() {
            self.maybe_storage_dead.seek_after_primary_effect(location);
            if self.maybe_storage_dead.get().contains(local) {
                self.fail(location, format!("use of local {local:?}, which has no storage here"));
            }
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(box (dest, rvalue)) => {
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) = rvalue {
                    // The sides of an assignment must not alias. Currently this just checks whether
                    // the places are identical.
                    if dest == src {
                        self.fail(
                            location,
                            "encountered `Assign` statement with overlapping memory",
                        );
                    }
                }
            }
            StatementKind::StorageLive(local) => {
                self.maybe_storage_live.seek_before_primary_effect(location);
                if self.maybe_storage_live.get().contains(*local) {
                    self.fail(
                        location,
                        format!("StorageLive({local:?}) which already has storage here"),
                    );
                }
            }
            _ => {}
        }

        self.super_statement(statement, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Return => {
                if self.is_fn_like {
                    self.maybe_storage_live.seek_after_primary_effect(location);
                    for local in self.maybe_storage_live.get().iter() {
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
            TerminatorKind::Call { args, destination, .. } => {
                // The call destination place and Operand::Move place used as an argument might be
                // passed by a reference to the callee. Consequently they must be non-overlapping.
                // Currently this simply checks for duplicate places.
                self.places.clear();
                self.places.insert(destination.as_ref());
                let mut has_duplicates = false;
                for arg in args {
                    if let Operand::Move(place) = &arg.node {
                        has_duplicates |= !self.places.insert(place.as_ref());
                    }
                }
                if has_duplicates {
                    self.fail(
                        location,
                        format!(
                            "encountered overlapping memory in `Move` arguments to `Call` terminator: {:?}",
                            terminator.kind,
                        ),
                    );
                }
            }
            _ => {}
        }

        self.super_terminator(terminator, location);
    }
}
