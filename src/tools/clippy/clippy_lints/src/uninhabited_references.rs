use clippy_utils::diagnostics::span_lint;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, Expr, ExprKind, FnDecl, FnRetTy, TyKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// It detects references to uninhabited types, such as `!` and
    /// warns when those are either dereferenced or returned from a function.
    ///
    /// ### Why is this bad?
    /// Dereferencing a reference to an uninhabited type would create
    /// an instance of such a type, which cannot exist. This constitutes
    /// undefined behaviour. Such a reference could have been created
    /// by `unsafe` code.
    ///
    /// ### Example
    /// The following function can return a reference to an uninhabited type
    /// (`Infallible`) because it uses `unsafe` code to create it. However,
    /// the user of such a function could dereference the return value and
    /// trigger an undefined behavior from safe code.
    ///
    /// ```no_run
    /// fn create_ref() -> &'static std::convert::Infallible {
    ///     unsafe { std::mem::transmute(&()) }
    /// }
    /// ```
    #[clippy::version = "1.76.0"]
    pub UNINHABITED_REFERENCES,
    nursery,
    "reference to uninhabited type"
}

declare_lint_pass!(UninhabitedReferences => [UNINHABITED_REFERENCES]);

impl LateLintPass<'_> for UninhabitedReferences {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Unary(UnOp::Deref, _) = expr.kind
            && cx
                .typeck_results
                .expr_ty_adjusted(expr)
                .is_privately_uninhabited(cx.tcx, cx.typing_env())
            && !expr.span.in_external_macro(cx.tcx.sess.source_map())
        {
            span_lint(
                cx,
                UNINHABITED_REFERENCES,
                expr.span,
                "dereferencing a reference to an uninhabited type is undefined behavior",
            );
        }
    }

    fn check_fn<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'_>,
        fndecl: &'_ FnDecl<'tcx>,
        _: &'_ Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if !matches!(kind, FnKind::Closure)
            && let FnRetTy::Return(hir_ty) = fndecl.output
            && let TyKind::Ref(..) = hir_ty.kind
            && let sig = cx.tcx.fn_sig(def_id).instantiate_identity()
            && let ty = sig.map(|x| cx.tcx.instantiate_bound_regions_with_erased(x.output()))
            && let Ok(ty) = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ty)
            && let ty::Ref(_, ty, _) = *ty.kind()
            && ty.is_privately_uninhabited(cx.tcx, cx.typing_env())
            && !span.in_external_macro(cx.tcx.sess.source_map())
        {
            span_lint(
                cx,
                UNINHABITED_REFERENCES,
                hir_ty.span,
                "dereferencing a reference to an uninhabited type would be undefined behavior",
            );
        }
    }
}
