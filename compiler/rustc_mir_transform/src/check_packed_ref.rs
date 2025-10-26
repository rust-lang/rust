use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::ty::{self, TyCtxt};

use crate::{errors, util};

pub(super) struct CheckPackedRef;

impl<'tcx> crate::MirLint<'tcx> for CheckPackedRef {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let source_info = SourceInfo::outermost(body.span);
        let mut checker = PackedRefChecker { body, tcx, typing_env, source_info };
        checker.visit_body(body);
    }
}

struct PackedRefChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    source_info: SourceInfo,
}

impl<'tcx> Visitor<'tcx> for PackedRefChecker<'_, 'tcx> {
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
        if context.is_borrow() && util::is_disaligned(self.tcx, self.body, self.typing_env, *place)
        {
            let def_id = self.body.source.instance.def_id();
            if let Some(impl_def_id) = self.tcx.trait_impl_of_assoc(def_id)
                && self.tcx.is_builtin_derived(impl_def_id)
            {
                // If we ever reach here it means that the generated derive
                // code is somehow doing an unaligned reference, which it
                // shouldn't do.
                span_bug!(self.source_info.span, "builtin derive created an unaligned reference");
            } else {
                self.tcx.dcx().emit_err(errors::UnalignedPackedRef { span: self.source_info.span });
            }
        }
    }
}
