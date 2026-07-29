use rustc_hir::attrs::lang_items::LangItem;
use rustc_lint_defs::builtin::{C_VOID_REFERENCES, C_VOID_VALUES};
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};

use crate::diagnostics;

pub(super) struct CheckCVoid;

impl<'tcx> crate::MirLint<'tcx> for CheckCVoid {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let source_info = SourceInfo::outermost(body.span);
        let mut checker = CVoidChecker { body, tcx, source_info };
        checker.visit_body(body);
    }
}

struct CVoidChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    source_info: SourceInfo,
}

impl<'tcx> Visitor<'tcx> for CVoidChecker<'_, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Make sure we know where in the MIR we are.
        self.source_info = terminator.source_info;
        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Make sure we know where in the MIR we are.
        self.source_info = statement.source_info;
        self.super_statement(statement, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        match context {
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Move | NonMutatingUseContext::SharedBorrow,
            )
            | PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Call
                | MutatingUseContext::Yield
                | MutatingUseContext::Borrow,
            ) => (),
            _ => return,
        };

        let ty = place.ty(self.body, self.tcx).ty;

        if is_c_void(self.tcx, ty.peel_refs()) {
            let Some(hir_id) = self.source_info.scope.lint_root(&self.body.source_scopes) else {
                return;
            };

            if context.is_borrow() || ty.is_ref() {
                self.tcx.emit_node_span_lint(
                    C_VOID_REFERENCES,
                    hir_id,
                    self.source_info.span,
                    diagnostics::CVoidRef,
                );
            } else {
                self.tcx.emit_node_span_lint(
                    C_VOID_VALUES,
                    hir_id,
                    self.source_info.span,
                    diagnostics::CVoidValue,
                );
            }
        }
    }
}

fn is_c_void(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    if let Some(adt_def) = ty.ty_adt_def() {
        tcx.is_lang_item(adt_def.did(), LangItem::CVoid)
    } else {
        false
    }
}
