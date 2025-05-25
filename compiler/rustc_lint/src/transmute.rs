use rustc_hir::def::{DefKind, Res};
use rustc_hir::{self as hir};
use rustc_macros::LintDiagnostic;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::sym;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `ptr_to_integer_transmute_in_consts` lint detects pointer to integer
    /// transmute in const functions and associated constants.
    ///
    /// ### Example
    ///
    /// ```rust
    /// const fn foo(ptr: *const u8) -> usize {
    ///    unsafe {
    ///        std::mem::transmute::<*const u8, usize>(ptr)
    ///    }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Transmuting pointers to integers in a `const` context is undefined behavior.
    /// Any attempt to use the resulting integer will abort const-evaluation.
    ///
    /// But sometimes the compiler might not emit an error for pointer to integer transmutes
    /// inside const functions and associated consts because they are evaluated only when referenced.
    /// Therefore, this lint serves as an extra layer of defense to prevent any undefined behavior
    /// from compiling without any warnings or errors.
    ///
    /// See [std::mem::transmute] in the reference for more details.
    ///
    /// [std::mem::transmute]: https://doc.rust-lang.org/std/mem/fn.transmute.html
    pub PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS,
    Warn,
    "detects pointer to integer transmutes in const functions and associated constants",
}

pub(crate) struct CheckTransmutes;

impl_lint_pass!(CheckTransmutes => [PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS]);

impl<'tcx> LateLintPass<'tcx> for CheckTransmutes {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Call(callee, _) = expr.kind else {
            return;
        };
        let hir::ExprKind::Path(qpath) = callee.kind else {
            return;
        };
        let Res::Def(DefKind::Fn, def_id) = cx.qpath_res(&qpath, callee.hir_id) else {
            return;
        };
        if !cx.tcx.is_intrinsic(def_id, sym::transmute) {
            return;
        };
        let body_owner_def_id = cx.tcx.hir_enclosing_body_owner(expr.hir_id);
        let Some(context) = cx.tcx.hir_body_const_context(body_owner_def_id) else {
            return;
        };
        let args = cx.typeck_results().node_args(callee.hir_id);

        let src = args.type_at(0);
        let dst = args.type_at(1);

        // Check for transmutes that exhibit undefined behavior.
        // For example, transmuting pointers to integers in a const context.
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
        if matches!(context, hir::ConstContext::ConstFn)
            || matches!(cx.tcx.def_kind(body_owner_def_id), DefKind::AssocConst)
        {
            if src.is_raw_ptr() && dst.is_integral() {
                cx.tcx.emit_node_span_lint(
                    PTR_TO_INTEGER_TRANSMUTE_IN_CONSTS,
                    expr.hir_id,
                    expr.span,
                    UndefinedTransmuteLint,
                );
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_undefined_transmute)]
#[note]
#[note(lint_note2)]
#[help]
pub(crate) struct UndefinedTransmuteLint;
