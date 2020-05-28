//! Validates the MIR to ensure that invariants are upheld.

use super::{MirPass, MirSource};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::{
    mir::{Body, Location, Operand, Rvalue, Statement, StatementKind},
    ty::{ParamEnv, TyCtxt},
};
use rustc_span::{def_id::DefId, Span, DUMMY_SP};

pub struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
}

impl<'tcx> MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let def_id = source.def_id();
        let param_env = tcx.param_env(def_id);
        TypeChecker { when: &self.when, def_id, body, tcx, param_env }.visit_body(body);
    }
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    def_id: DefId,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
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

            if !ty.is_copy_modulo_regions(self.tcx, self.param_env, DUMMY_SP) {
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
    }
}
