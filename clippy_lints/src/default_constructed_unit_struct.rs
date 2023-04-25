use clippy_utils::{diagnostics::span_lint_and_sugg, is_from_proc_macro, match_def_path, paths};
use hir::{def::Res, ExprKind};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Check for construction on unit struct using `default`.
    ///
    /// ### Why is this bad?
    /// This adds code complexity and an unnecessary function call.
    ///
    /// ### Example
    /// ```rust
    /// #[derive(Default)]
    /// struct S<T> {
    ///     _marker: PhantomData<T>
    /// }
    ///
    /// let _: S<i32> = S {
    ///     _marker: PhantomData::default()
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// let _: S<i32> = Something {
    ///     _marker: PhantomData
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub DEFAULT_CONSTRUCTED_UNIT_STRUCT,
    complexity,
    "unit structs can be contructed without calling `default`"
}
declare_lint_pass!(DefaultConstructedUnitStruct => [DEFAULT_CONSTRUCTED_UNIT_STRUCT]);

impl LateLintPass<'_> for DefaultConstructedUnitStruct {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if_chain!(
            // make sure we have a call to `Default::default`
            if let hir::ExprKind::Call(fn_expr, &[]) = expr.kind;
            if let ExprKind::Path(ref qpath) = fn_expr.kind;
            if let Res::Def(_, def_id) = cx.qpath_res(qpath, fn_expr.hir_id);
            if match_def_path(cx, def_id, &paths::DEFAULT_TRAIT_METHOD);
            // make sure we have a struct with no fields (unit struct)
            if let ty::Adt(def, ..) = cx.typeck_results().expr_ty(expr).kind();
            if def.is_struct() && def.is_payloadfree()
                && !def.non_enum_variant().is_field_list_non_exhaustive()
                && !is_from_proc_macro(cx, expr);
            then {
                span_lint_and_sugg(
                    cx,
                    DEFAULT_CONSTRUCTED_UNIT_STRUCT,
                    qpath.last_segment_span(),
                    "Use of `default` to create a unit struct.",
                    "remove this call to `default`",
                    String::new(),
                    Applicability::MachineApplicable,
                )
            }
        );
    }
}
