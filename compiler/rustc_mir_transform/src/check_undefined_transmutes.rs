use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{Body, Location, Operand, Terminator, TerminatorKind};
use rustc_middle::ty::{AssocItem, AssocKind, TyCtxt};
use rustc_session::lint::builtin::PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS;
use rustc_span::sym;

use crate::errors;

/// Check for transmutes that exhibit undefined behavior.
/// For example, transmuting pointers to integers in a const context.
pub(super) struct CheckUndefinedTransmutes;

impl<'tcx> crate::MirLint<'tcx> for CheckUndefinedTransmutes {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let mut checker = UndefinedTransmutesChecker { body, tcx };
        checker.visit_body(body);
    }
}

struct UndefinedTransmutesChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> UndefinedTransmutesChecker<'a, 'tcx> {
    // This functions checks two things:
    // 1. `function` takes a raw pointer as input and returns an integer as output.
    // 2. `function` is called from a const function or an associated constant.
    //
    // Why do we consider const functions and associated constants only?
    //
    // Generally, undefined behavior in const items are handled by the evaluator.
    // But, const functions and associated constants are evaluated only when referenced.
    // This can result in undefined behavior in a library going unnoticed until
    // the function or constant is actually used.
    //
    // Therefore, we only consider const functions and associated constants here and leave
    // other const items to be handled by the evaluator.
    fn is_ptr_to_int_in_const(&self, function: &Operand<'tcx>) -> bool {
        let def_id = self.body.source.def_id();

        if self.tcx.is_const_fn(def_id)
            || matches!(
                self.tcx.opt_associated_item(def_id),
                Some(AssocItem { kind: AssocKind::Const { .. }, .. })
            )
        {
            let fn_sig = function.ty(self.body, self.tcx).fn_sig(self.tcx).skip_binder();
            if let [input] = fn_sig.inputs() {
                return input.is_raw_ptr() && fn_sig.output().is_integral();
            }
        }
        false
    }
}

impl<'tcx> Visitor<'tcx> for UndefinedTransmutesChecker<'_, 'tcx> {
    // Check each block's terminator for calls to pointer to integer transmutes
    // in const functions or associated constants and emit a lint.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call { func, .. } = &terminator.kind
            && let Some((func_def_id, _)) = func.const_fn_def()
            && self.tcx.is_intrinsic(func_def_id, sym::transmute)
            && self.is_ptr_to_int_in_const(func)
            && let Some(call_id) = self.body.source.def_id().as_local()
        {
            let hir_id = self.tcx.local_def_id_to_hir_id(call_id);
            let span = self.body.source_info(location).span;
            self.tcx.emit_node_span_lint(
                PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS,
                hir_id,
                span,
                errors::UndefinedTransmute,
            );
        }
    }
}
