use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_from_proc_macro;
use clippy_utils::ty::needs_ordered_drop;
use rustc_ast::Mutability;
use rustc_hir::def::Res;
use rustc_hir::{BindingAnnotation, ByRef, ExprKind, HirId, Local, Node, Pat, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Ident;
use rustc_span::DesugaringKind;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for redundant redefinitions of local bindings.
    ///
    /// ### Why is this bad?
    /// Redundant redefinitions of local bindings do not change behavior and are likely to be unintended.
    ///
    /// Note that although these bindings do not affect your code's meaning, they _may_ affect `rustc`'s stack allocation.
    ///
    /// ### Example
    /// ```no_run
    /// let a = 0;
    /// let a = a;
    ///
    /// fn foo(b: i32) {
    ///    let b = b;
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a = 0;
    /// // no redefinition with the same name
    ///
    /// fn foo(b: i32) {
    ///   // no redefinition with the same name
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub REDUNDANT_LOCALS,
    correctness,
    "redundant redefinition of a local binding"
}
declare_lint_pass!(RedundantLocals => [REDUNDANT_LOCALS]);

impl<'tcx> LateLintPass<'tcx> for RedundantLocals {
    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &'tcx Local<'tcx>) {
        if !local.span.is_desugaring(DesugaringKind::Async)
            // the pattern is a single by-value binding
            && let PatKind::Binding(BindingAnnotation(ByRef::No, mutability), _, ident, None) = local.pat.kind
            // the binding is not type-ascribed
            && local.ty.is_none()
            // the expression is a resolved path
            && let Some(expr) = local.init
            && let ExprKind::Path(qpath @ QPath::Resolved(None, path)) = expr.kind
            // the path is a single segment equal to the local's name
            && let [last_segment] = path.segments
            && last_segment.ident == ident
            // resolve the path to its defining binding pattern
            && let Res::Local(binding_id) = cx.qpath_res(&qpath, expr.hir_id)
            && let Node::Pat(binding_pat) = cx.tcx.hir().get(binding_id)
            // the previous binding has the same mutability
            && find_binding(binding_pat, ident).is_some_and(|bind| bind.1 == mutability)
            // the local does not change the effect of assignments to the binding. see #11290
            && !affects_assignments(cx, mutability, binding_id, local.hir_id)
            // the local does not affect the code's drop behavior
            && !needs_ordered_drop(cx, cx.typeck_results().expr_ty(expr))
            // the local is user-controlled
            && !in_external_macro(cx.sess(), local.span)
            && !is_from_proc_macro(cx, expr)
        {
            span_lint_and_help(
                cx,
                REDUNDANT_LOCALS,
                local.span,
                &format!("redundant redefinition of a binding `{ident}`"),
                Some(binding_pat.span),
                &format!("`{ident}` is initially defined here"),
            );
        }
    }
}

/// Find the annotation of a binding introduced by a pattern, or `None` if it's not introduced.
fn find_binding(pat: &Pat<'_>, name: Ident) -> Option<BindingAnnotation> {
    let mut ret = None;

    pat.each_binding_or_first(&mut |annotation, _, _, ident| {
        if ident == name {
            ret = Some(annotation);
        }
    });

    ret
}

/// Check if a rebinding of a local changes the effect of assignments to the binding.
fn affects_assignments(cx: &LateContext<'_>, mutability: Mutability, bind: HirId, rebind: HirId) -> bool {
    let hir = cx.tcx.hir();

    // the binding is mutable and the rebinding is in a different scope than the original binding
    mutability == Mutability::Mut && hir.get_enclosing_scope(bind) != hir.get_enclosing_scope(rebind)
}
