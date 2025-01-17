use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::{
    SpanlessEq, higher, is_in_const_context, is_integer_literal, path_to_local, peel_blocks, peel_blocks_with_stmt,
};
use rustc_ast::ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{BinOp, BinOpKind, Expr, ExprKind, HirId, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for implicit saturating subtraction.
    ///
    /// ### Why is this bad?
    /// Simplicity and readability. Instead we can easily use an builtin function.
    ///
    /// ### Example
    /// ```no_run
    /// # let end: u32 = 10;
    /// # let start: u32 = 5;
    /// let mut i: u32 = end - start;
    ///
    /// if i != 0 {
    ///     i -= 1;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let end: u32 = 10;
    /// # let start: u32 = 5;
    /// let mut i: u32 = end - start;
    ///
    /// i = i.saturating_sub(1);
    /// ```
    #[clippy::version = "1.44.0"]
    pub IMPLICIT_SATURATING_SUB,
    style,
    "Perform saturating subtraction instead of implicitly checking lower bound of data type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for comparisons between integers, followed by subtracting the greater value from the
    /// lower one.
    ///
    /// ### Why is this bad?
    /// This could result in an underflow and is most likely not what the user wants. If this was
    /// intended to be a saturated subtraction, consider using the `saturating_sub` method directly.
    ///
    /// ### Example
    /// ```no_run
    /// let a = 12u32;
    /// let b = 13u32;
    ///
    /// let result = if a > b { b - a } else { 0 };
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a = 12u32;
    /// let b = 13u32;
    ///
    /// let result = a.saturating_sub(b);
    /// ```
    #[clippy::version = "1.44.0"]
    pub INVERTED_SATURATING_SUB,
    correctness,
    "Check if a variable is smaller than another one and still subtract from it even if smaller"
}

pub struct ImplicitSaturatingSub {
    msrv: Msrv,
}

impl_lint_pass!(ImplicitSaturatingSub => [IMPLICIT_SATURATING_SUB, INVERTED_SATURATING_SUB]);

impl ImplicitSaturatingSub {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv.clone(),
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ImplicitSaturatingSub {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if expr.span.from_expansion() {
            return;
        }
        if let Some(higher::If { cond, then, r#else: None }) = higher::If::hir(expr)

            // Check if the conditional expression is a binary operation
            && let ExprKind::Binary(ref cond_op, cond_left, cond_right) = cond.kind
        {
            check_with_condition(cx, expr, cond_op.node, cond_left, cond_right, then);
        } else if let Some(higher::If {
            cond,
            then: if_block,
            r#else: Some(else_block),
        }) = higher::If::hir(expr)
            && let ExprKind::Binary(ref cond_op, cond_left, cond_right) = cond.kind
        {
            check_manual_check(
                cx, expr, cond_op, cond_left, cond_right, if_block, else_block, &self.msrv,
            );
        }
    }

    extract_msrv_attr!(LateContext);
}

#[allow(clippy::too_many_arguments)]
fn check_manual_check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    condition: &BinOp,
    left_hand: &Expr<'tcx>,
    right_hand: &Expr<'tcx>,
    if_block: &Expr<'tcx>,
    else_block: &Expr<'tcx>,
    msrv: &Msrv,
) {
    let ty = cx.typeck_results().expr_ty(left_hand);
    if ty.is_numeric() && !ty.is_signed() {
        match condition.node {
            BinOpKind::Gt | BinOpKind::Ge => check_gt(
                cx,
                condition.span,
                expr.span,
                left_hand,
                right_hand,
                if_block,
                else_block,
                msrv,
                matches!(
                    clippy_utils::get_parent_expr(cx, expr),
                    Some(Expr {
                        kind: ExprKind::If(..),
                        ..
                    })
                ),
            ),
            BinOpKind::Lt | BinOpKind::Le => check_gt(
                cx,
                condition.span,
                expr.span,
                right_hand,
                left_hand,
                if_block,
                else_block,
                msrv,
                matches!(
                    clippy_utils::get_parent_expr(cx, expr),
                    Some(Expr {
                        kind: ExprKind::If(..),
                        ..
                    })
                ),
            ),
            _ => {},
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_gt(
    cx: &LateContext<'_>,
    condition_span: Span,
    expr_span: Span,
    big_var: &Expr<'_>,
    little_var: &Expr<'_>,
    if_block: &Expr<'_>,
    else_block: &Expr<'_>,
    msrv: &Msrv,
    is_composited: bool,
) {
    if let Some(big_var) = Var::new(big_var)
        && let Some(little_var) = Var::new(little_var)
    {
        check_subtraction(
            cx,
            condition_span,
            expr_span,
            big_var,
            little_var,
            if_block,
            else_block,
            msrv,
            is_composited,
        );
    }
}

struct Var {
    span: Span,
    hir_id: HirId,
}

impl Var {
    fn new(expr: &Expr<'_>) -> Option<Self> {
        path_to_local(expr).map(|hir_id| Self {
            span: expr.span,
            hir_id,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn check_subtraction(
    cx: &LateContext<'_>,
    condition_span: Span,
    expr_span: Span,
    big_var: Var,
    little_var: Var,
    if_block: &Expr<'_>,
    else_block: &Expr<'_>,
    msrv: &Msrv,
    is_composited: bool,
) {
    let if_block = peel_blocks(if_block);
    let else_block = peel_blocks(else_block);
    if is_integer_literal(if_block, 0) {
        // We need to check this case as well to prevent infinite recursion.
        if is_integer_literal(else_block, 0) {
            // Well, seems weird but who knows?
            return;
        }
        // If the subtraction is done in the `else` block, then we need to also revert the two
        // variables as it means that the check was reverted too.
        check_subtraction(
            cx,
            condition_span,
            expr_span,
            little_var,
            big_var,
            else_block,
            if_block,
            msrv,
            is_composited,
        );
        return;
    }
    if is_integer_literal(else_block, 0)
        && let ExprKind::Binary(op, left, right) = if_block.kind
        && let BinOpKind::Sub = op.node
    {
        let local_left = path_to_local(left);
        let local_right = path_to_local(right);
        if Some(big_var.hir_id) == local_left && Some(little_var.hir_id) == local_right {
            // This part of the condition is voluntarily split from the one before to ensure that
            // if `snippet_opt` fails, it won't try the next conditions.
            if let Some(big_var_snippet) = snippet_opt(cx, big_var.span)
                && let Some(little_var_snippet) = snippet_opt(cx, little_var.span)
                && (!is_in_const_context(cx) || msrv.meets(msrvs::SATURATING_SUB_CONST))
            {
                let sugg = format!(
                    "{}{big_var_snippet}.saturating_sub({little_var_snippet}){}",
                    if is_composited { "{ " } else { "" },
                    if is_composited { " }" } else { "" }
                );
                span_lint_and_sugg(
                    cx,
                    IMPLICIT_SATURATING_SUB,
                    expr_span,
                    "manual arithmetic check found",
                    "replace it with",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        } else if Some(little_var.hir_id) == local_left
            && Some(big_var.hir_id) == local_right
            && let Some(big_var_snippet) = snippet_opt(cx, big_var.span)
            && let Some(little_var_snippet) = snippet_opt(cx, little_var.span)
        {
            span_lint_and_then(
                cx,
                INVERTED_SATURATING_SUB,
                condition_span,
                "inverted arithmetic check before subtraction",
                |diag| {
                    diag.span_note(
                        if_block.span,
                        format!("this subtraction underflows when `{little_var_snippet} < {big_var_snippet}`"),
                    );
                    diag.span_suggestion(
                        if_block.span,
                        "try replacing it with",
                        format!("{big_var_snippet} - {little_var_snippet}"),
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}

fn check_with_condition<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cond_op: BinOpKind,
    cond_left: &Expr<'tcx>,
    cond_right: &Expr<'tcx>,
    then: &Expr<'tcx>,
) {
    // Ensure that the binary operator is >, !=, or <
    if (BinOpKind::Ne == cond_op || BinOpKind::Gt == cond_op || BinOpKind::Lt == cond_op)

        // Check if assign operation is done
        && let Some(target) = subtracts_one(cx, then)

        // Extracting out the variable name
        && let ExprKind::Path(QPath::Resolved(_, ares_path)) = target.kind
    {
        // Handle symmetric conditions in the if statement
        let (cond_var, cond_num_val) = if SpanlessEq::new(cx).eq_expr(cond_left, target) {
            if BinOpKind::Gt == cond_op || BinOpKind::Ne == cond_op {
                (cond_left, cond_right)
            } else {
                return;
            }
        } else if SpanlessEq::new(cx).eq_expr(cond_right, target) {
            if BinOpKind::Lt == cond_op || BinOpKind::Ne == cond_op {
                (cond_right, cond_left)
            } else {
                return;
            }
        } else {
            return;
        };

        // Check if the variable in the condition statement is an integer
        if !cx.typeck_results().expr_ty(cond_var).is_integral() {
            return;
        }

        // Get the variable name
        let var_name = ares_path.segments[0].ident.name.as_str();
        match cond_num_val.kind {
            ExprKind::Lit(cond_lit) => {
                // Check if the constant is zero
                if let LitKind::Int(Pu128(0), _) = cond_lit.node {
                    if cx.typeck_results().expr_ty(cond_left).is_signed() {
                    } else {
                        print_lint_and_sugg(cx, var_name, expr);
                    };
                }
            },
            ExprKind::Path(QPath::TypeRelative(_, name)) => {
                if name.ident.as_str() == "MIN"
                    && let Some(const_id) = cx.typeck_results().type_dependent_def_id(cond_num_val.hir_id)
                    && let Some(impl_id) = cx.tcx.impl_of_method(const_id)
                    && let None = cx.tcx.impl_trait_ref(impl_id) // An inherent impl
                    && cx.tcx.type_of(impl_id).instantiate_identity().is_integral()
                {
                    print_lint_and_sugg(cx, var_name, expr);
                }
            },
            ExprKind::Call(func, []) => {
                if let ExprKind::Path(QPath::TypeRelative(_, name)) = func.kind
                    && name.ident.as_str() == "min_value"
                    && let Some(func_id) = cx.typeck_results().type_dependent_def_id(func.hir_id)
                    && let Some(impl_id) = cx.tcx.impl_of_method(func_id)
                    && let None = cx.tcx.impl_trait_ref(impl_id) // An inherent impl
                    && cx.tcx.type_of(impl_id).instantiate_identity().is_integral()
                {
                    print_lint_and_sugg(cx, var_name, expr);
                }
            },
            _ => (),
        }
    }
}

fn subtracts_one<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<&'a Expr<'a>> {
    match peel_blocks_with_stmt(expr).kind {
        ExprKind::AssignOp(ref op1, target, value) => {
            // Check if literal being subtracted is one
            (BinOpKind::Sub == op1.node && is_integer_literal(value, 1)).then_some(target)
        },
        ExprKind::Assign(target, value, _) => {
            if let ExprKind::Binary(ref op1, left1, right1) = value.kind
                && BinOpKind::Sub == op1.node
                && SpanlessEq::new(cx).eq_expr(left1, target)
                && is_integer_literal(right1, 1)
            {
                Some(target)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn print_lint_and_sugg(cx: &LateContext<'_>, var_name: &str, expr: &Expr<'_>) {
    span_lint_and_sugg(
        cx,
        IMPLICIT_SATURATING_SUB,
        expr.span,
        "implicitly performing saturating subtraction",
        "try",
        format!("{var_name} = {var_name}.saturating_sub({});", '1'),
        Applicability::MachineApplicable,
    );
}
