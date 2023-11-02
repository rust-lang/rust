use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons with an address of a function item.
    ///
    /// ### Why is this bad?
    /// Function item address is not guaranteed to be unique and could vary
    /// between different code generation units. Furthermore different function items could have
    /// the same address after being merged together.
    ///
    /// ### Example
    /// ```no_run
    /// type F = fn();
    /// fn a() {}
    /// let f: F = a;
    /// if f == a {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "1.44.0"]
    pub FN_ADDRESS_COMPARISONS,
    correctness,
    "comparison with an address of a function item"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons with an address of a trait vtable.
    ///
    /// ### Why is this bad?
    /// Comparing trait objects pointers compares an vtable addresses which
    /// are not guaranteed to be unique and could vary between different code generation units.
    /// Furthermore vtables for different types could have the same address after being merged
    /// together.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a: Rc<dyn Trait> = ...
    /// let b: Rc<dyn Trait> = ...
    /// if Rc::ptr_eq(&a, &b) {
    ///     ...
    /// }
    /// ```
    #[clippy::version = "1.44.0"]
    pub VTABLE_ADDRESS_COMPARISONS,
    correctness,
    "comparison with an address of a trait vtable"
}

declare_lint_pass!(UnnamedAddress => [FN_ADDRESS_COMPARISONS, VTABLE_ADDRESS_COMPARISONS]);

impl LateLintPass<'_> for UnnamedAddress {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        fn is_comparison(binop: BinOpKind) -> bool {
            matches!(
                binop,
                BinOpKind::Eq | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Ne | BinOpKind::Ge | BinOpKind::Gt
            )
        }

        fn is_trait_ptr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            match cx.typeck_results().expr_ty_adjusted(expr).kind() {
                ty::RawPtr(ty::TypeAndMut { ty, .. }) => ty.is_trait(),
                _ => false,
            }
        }

        fn is_fn_def(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
            matches!(cx.typeck_results().expr_ty(expr).kind(), ty::FnDef(..))
        }

        if let ExprKind::Binary(binop, left, right) = expr.kind
            && is_comparison(binop.node)
            && is_trait_ptr(cx, left)
            && is_trait_ptr(cx, right)
        {
            span_lint_and_help(
                cx,
                VTABLE_ADDRESS_COMPARISONS,
                expr.span,
                "comparing trait object pointers compares a non-unique vtable address",
                None,
                "consider extracting and comparing data pointers only",
            );
        }

        if let ExprKind::Call(func, [ref _left, ref _right]) = expr.kind
            && let ExprKind::Path(ref func_qpath) = func.kind
            && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::ptr_eq, def_id)
            && let ty_param = cx.typeck_results().node_args(func.hir_id).type_at(0)
            && ty_param.is_trait()
        {
            span_lint_and_help(
                cx,
                VTABLE_ADDRESS_COMPARISONS,
                expr.span,
                "comparing trait object pointers compares a non-unique vtable address",
                None,
                "consider extracting and comparing data pointers only",
            );
        }

        if let ExprKind::Binary(binop, left, right) = expr.kind
            && is_comparison(binop.node)
            && cx.typeck_results().expr_ty_adjusted(left).is_fn_ptr()
            && cx.typeck_results().expr_ty_adjusted(right).is_fn_ptr()
            && (is_fn_def(cx, left) || is_fn_def(cx, right))
        {
            span_lint(
                cx,
                FN_ADDRESS_COMPARISONS,
                expr.span,
                "comparing with a non-unique address of a function item",
            );
        }
    }
}
