use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{match_def_path, paths, sugg};
use if_chain::if_chain;
use rustc_ast::util::parser::AssocOp;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for statements of the form `(a - b) < f32::EPSILON` or
    /// `(a - b) < f64::EPSILON`. Notes the missing `.abs()`.
    ///
    /// ### Why is this bad?
    /// The code without `.abs()` is more likely to have a bug.
    ///
    /// ### Known problems
    /// If the user can ensure that b is larger than a, the `.abs()` is
    /// technically unneccessary. However, it will make the code more robust and doesn't have any
    /// large performance implications. If the abs call was deliberately left out for performance
    /// reasons, it is probably better to state this explicitly in the code, which then can be done
    /// with an allow.
    ///
    /// ### Example
    /// ```rust
    /// pub fn is_roughly_equal(a: f32, b: f32) -> bool {
    ///     (a - b) < f32::EPSILON
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// pub fn is_roughly_equal(a: f32, b: f32) -> bool {
    ///     (a - b).abs() < f32::EPSILON
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub FLOAT_EQUALITY_WITHOUT_ABS,
    suspicious,
    "float equality check without `.abs()`"
}

declare_lint_pass!(FloatEqualityWithoutAbs => [FLOAT_EQUALITY_WITHOUT_ABS]);

impl<'tcx> LateLintPass<'tcx> for FloatEqualityWithoutAbs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let lhs;
        let rhs;

        // check if expr is a binary expression with a lt or gt operator
        if let ExprKind::Binary(op, left, right) = expr.kind {
            match op.node {
                BinOpKind::Lt => {
                    lhs = left;
                    rhs = right;
                },
                BinOpKind::Gt => {
                    lhs = right;
                    rhs = left;
                },
                _ => return,
            };
        } else {
            return;
        }

        if_chain! {

            // left hand side is a substraction
            if let ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Sub,
                    ..
                },
                val_l,
                val_r,
            ) = lhs.kind;

            // right hand side matches either f32::EPSILON or f64::EPSILON
            if let ExprKind::Path(ref epsilon_path) = rhs.kind;
            if let Res::Def(DefKind::AssocConst, def_id) = cx.qpath_res(epsilon_path, rhs.hir_id);
            if match_def_path(cx, def_id, &paths::F32_EPSILON) || match_def_path(cx, def_id, &paths::F64_EPSILON);

            // values of the substractions on the left hand side are of the type float
            let t_val_l = cx.typeck_results().expr_ty(val_l);
            let t_val_r = cx.typeck_results().expr_ty(val_r);
            if let ty::Float(_) = t_val_l.kind();
            if let ty::Float(_) = t_val_r.kind();

            then {
                let sug_l = sugg::Sugg::hir(cx, val_l, "..");
                let sug_r = sugg::Sugg::hir(cx, val_r, "..");
                // format the suggestion
                let suggestion = format!("{}.abs()", sugg::make_assoc(AssocOp::Subtract, &sug_l, &sug_r).maybe_par());
                // spans the lint
                span_lint_and_then(
                    cx,
                    FLOAT_EQUALITY_WITHOUT_ABS,
                    expr.span,
                    "float equality check without `.abs()`",
                    | diag | {
                        diag.span_suggestion(
                            lhs.span,
                            "add `.abs()`",
                            suggestion,
                            Applicability::MaybeIncorrect,
                        );
                    }
                );
            }
        }
    }
}
