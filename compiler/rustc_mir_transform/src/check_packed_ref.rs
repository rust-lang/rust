use rustc_errors::struct_span_err;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

use crate::util;
use crate::MirLint;

pub struct CheckPackedRef;

impl<'tcx> MirLint<'tcx> for CheckPackedRef {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let param_env = tcx.param_env(body.source.def_id());
        let source_info = SourceInfo::outermost(body.span);
        let mut checker = PackedRefChecker { body, tcx, param_env, source_info };
        checker.visit_body(&body);
    }
}

struct PackedRefChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
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
        if context.is_borrow() {
            if util::is_disaligned(self.tcx, self.body, self.param_env, *place) {
                let def_id = self.body.source.instance.def_id();
                if let Some(impl_def_id) = self.tcx.impl_of_method(def_id)
                    && self.tcx.is_builtin_derive(impl_def_id)
                {
                    // If we ever reach here it means that the generated derive
                    // code is somehow doing an unaligned reference, which it
                    // shouldn't do.
                    unreachable!();
                } else {
                    struct_span_err!(
                        self.tcx.sess,
                        self.source_info.span,
                        E0793,
                        "reference to packed field is unaligned"
                    )
                    .note(
                        "fields of packed structs are not properly aligned, and creating \
                        a misaligned reference is undefined behavior (even if that \
                        reference is never dereferenced)",
                    ).help(
                        "copy the field contents to a local variable, or replace the \
                        reference with a raw pointer and use `read_unaligned`/`write_unaligned` \
                        (loads and stores via `*p` must be properly aligned even when using raw pointers)"
                    )
                    .emit();
                }
            }
        }
    }
}
