//! Validates the MIR to ensure that invariants are upheld.

use super::{MirPass, MirSource};
use crate::dataflow::{impls::MaybeInitializedLocals, Analysis, ResultsCursor};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::ty;
use rustc_middle::{
    mir::{traversal, Body, Local, Location, Operand, Rvalue, Statement, StatementKind},
    ty::{ParamEnv, TyCtxt},
};
use rustc_span::{def_id::DefId, Span, DUMMY_SP};
use ty::Ty;

pub struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
}

impl<'tcx> MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let def_id = source.def_id();
        let param_env = tcx.param_env(def_id);

        // Do not consider moves to deinitialize locals. Some MIR passes output MIR that violates
        // this assumption and would lead to uses of uninitialized data.
        let init = MaybeInitializedLocals::no_deinit_on_move()
            .into_engine(tcx, body, def_id)
            .iterate_to_fixpoint()
            .into_results_cursor(body);

        let mut checker = TypeChecker { when: &self.when, def_id, body, tcx, param_env, init };

        // Only visit reachable blocks. Unreachable code may access uninitialized locals.
        for (block, data) in traversal::preorder(body) {
            checker.visit_basic_block_data(block, data);
        }
    }
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,

    def_id: DefId,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,

    init: ResultsCursor<'a, 'tcx, MaybeInitializedLocals>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn fail(&self, span: Span, msg: impl AsRef<str>) {
        // We use `delay_span_bug` as we might see broken MIR when other errors have already
        // occurred.
        self.tcx.sess.diagnostic().delay_span_bug(
            span,
            &format!("broken MIR in {:?} ({}): {}", self.def_id, self.when, msg.as_ref()),
        );
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // `Operand::Copy` is only supposed to be used with `Copy` types.
        if let Operand::Copy(place) = operand {
            let ty = place.ty(&self.body.local_decls, self.tcx).ty;

            if false && !ty.is_copy_modulo_regions(self.tcx, self.param_env, DUMMY_SP) {
                self.fail(
                    DUMMY_SP,
                    format!("`Operand::Copy` with non-`Copy` type {} at {:?}", ty, location),
                );
            }
        }

        self.super_operand(operand, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // The sides of an assignment must not alias. Currently this just checks whether the places
        // are identical.
        if let StatementKind::Assign(box (dest, rvalue)) = &statement.kind {
            match rvalue {
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                    if dest == src {
                        self.fail(
                            DUMMY_SP,
                            format!(
                                "encountered `Assign` statement with overlapping memory at {:?}",
                                location
                            ),
                        );
                    }
                }
                _ => {}
            }
        }

        // Every local used by a statement must be initialized before the statement executes.
        self.init.seek_before_primary_effect(location);
        UsedLocalsAreInitialized {
            checker: self,
            init: self.init.get(),
            span: statement.source_info.span,
        }
        .visit_statement(statement, location);
    }
}

struct UsedLocalsAreInitialized<'a, 'tcx> {
    checker: &'a TypeChecker<'a, 'tcx>,
    init: &'a BitSet<Local>,
    span: Span,
}

impl Visitor<'tcx> for UsedLocalsAreInitialized<'a, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, location: Location) {
        if context.is_use() && !context.is_place_assignment() && !self.init.contains(*local) {
            if context == PlaceContext::MutatingUse(MutatingUseContext::Projection) {
                // Ignore `_1.b`-like projections as they appear as assignment destinations, and
                // `_1` doesn't have to be initialized there.
                return;
            }

            if is_zst(
                self.checker.tcx,
                self.checker.def_id,
                self.checker.body.local_decls[*local].ty,
            ) {
                // Known ZSTs don't have to be initialized at all, skip them.
                return;
            }

            self.checker.fail(
                self.span,
                format!("use of uninitialized local {:?} at {:?}", local, location),
            );
        }
    }
}

fn is_zst<'tcx>(tcx: TyCtxt<'tcx>, did: DefId, ty: Ty<'tcx>) -> bool {
    tcx.layout_of(tcx.param_env(did).and(ty)).map(|layout| layout.is_zst()).unwrap_or(false)
}
