use clippy_utils::consts::Constant::{F32, F64, Int};
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::{
    eq_expr_value, get_parent_expr, higher, is_in_const_context, is_inherent_method_call, is_no_std_crate,
    numeric_literal, peel_blocks, sugg, sym,
};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::source_map::Spanned;

use rustc_ast::ast;
use std::f32::consts as f32_consts;
use std::f64::consts as f64_consts;
use sugg::Sugg;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for floating-point expressions that
    /// can be expressed using built-in methods to improve accuracy
    /// at the cost of performance.
    ///
    /// ### Why is this bad?
    /// Negatively impacts accuracy.
    ///
    /// ### Example
    /// ```no_run
    /// let a = 3f32;
    /// let _ = a.powf(1.0 / 3.0);
    /// let _ = (1.0 + a).ln();
    /// let _ = a.exp() - 1.0;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a = 3f32;
    /// let _ = a.cbrt();
    /// let _ = a.ln_1p();
    /// let _ = a.exp_m1();
    /// ```
    #[clippy::version = "1.43.0"]
    pub IMPRECISE_FLOPS,
    nursery,
    "usage of imprecise floating point operations"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for floating-point expressions that
    /// can be expressed using built-in methods to improve both
    /// accuracy and performance.
    ///
    /// ### Why is this bad?
    /// Negatively impacts accuracy and performance.
    ///
    /// ### Example
    /// ```no_run
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = (2f32).powf(a);
    /// let _ = E.powf(a);
    /// let _ = a.powf(1.0 / 2.0);
    /// let _ = a.log(2.0);
    /// let _ = a.log(10.0);
    /// let _ = a.log(E);
    /// let _ = a.powf(2.0);
    /// let _ = a * 2.0 + 4.0;
    /// let _ = if a < 0.0 {
    ///     -a
    /// } else {
    ///     a
    /// };
    /// let _ = if a < 0.0 {
    ///     a
    /// } else {
    ///     -a
    /// };
    /// ```
    ///
    /// is better expressed as
    ///
    /// ```no_run
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = a.exp2();
    /// let _ = a.exp();
    /// let _ = a.sqrt();
    /// let _ = a.log2();
    /// let _ = a.log10();
    /// let _ = a.ln();
    /// let _ = a.powi(2);
    /// let _ = a.mul_add(2.0, 4.0);
    /// let _ = a.abs();
    /// let _ = -a.abs();
    /// ```
    #[clippy::version = "1.43.0"]
    pub SUBOPTIMAL_FLOPS,
    nursery,
    "usage of sub-optimal floating point operations"
}

declare_lint_pass!(FloatingPointArithmetic => [
    IMPRECISE_FLOPS,
    SUBOPTIMAL_FLOPS
]);

// Returns the specialized log method for a given base if base is constant
// and is one of 2, 10 and e
fn get_specialized_log_method(cx: &LateContext<'_>, base: &Expr<'_>) -> Option<&'static str> {
    if let Some(value) = ConstEvalCtxt::new(cx).eval(base) {
        if F32(2.0) == value || F64(2.0) == value {
            return Some("log2");
        } else if F32(10.0) == value || F64(10.0) == value {
            return Some("log10");
        } else if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            return Some("ln");
        }
    }

    None
}

// Adds type suffixes and parenthesis to method receivers if necessary
fn prepare_receiver_sugg<'a>(cx: &LateContext<'_>, mut expr: &'a Expr<'a>) -> Sugg<'a> {
    let mut suggestion = Sugg::hir(cx, expr, "..");

    if let ExprKind::Unary(UnOp::Neg, inner_expr) = &expr.kind {
        expr = inner_expr;
    }

    if let ty::Float(float_ty) = cx.typeck_results().expr_ty(expr).kind()
        // if the expression is a float literal and it is unsuffixed then
        // add a suffix so the suggestion is valid and unambiguous
        && let ExprKind::Lit(lit) = &expr.kind
        && let ast::LitKind::Float(sym, ast::LitFloatType::Unsuffixed) = lit.node
    {
        let op = format!(
            "{suggestion}{}{}",
            // Check for float literals without numbers following the decimal
            // separator such as `2.` and adds a trailing zero
            if sym.as_str().ends_with('.') { "0" } else { "" },
            float_ty.name_str()
        )
        .into();

        suggestion = match suggestion {
            Sugg::MaybeParen(_) => Sugg::MaybeParen(op),
            _ => Sugg::NonParen(op),
        };
    }

    suggestion.maybe_paren()
}

fn check_log_base(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    if let Some(method) = get_specialized_log_method(cx, &args[0]) {
        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "logarithm for bases 2, 10 and e can be computed more accurately",
            "consider using",
            format!("{}.{method}()", Sugg::hir(cx, receiver, "..").maybe_paren()),
            Applicability::MachineApplicable,
        );
    }
}

// TODO: Lint expressions of the form `(x + y).ln()` where y > 1 and
// suggest usage of `(x + (y - 1)).ln_1p()` instead
fn check_ln1p(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Add, ..
        },
        lhs,
        rhs,
    ) = receiver.kind
    {
        let ecx = ConstEvalCtxt::new(cx);
        let recv = match (ecx.eval(lhs), ecx.eval(rhs)) {
            (Some(value), _) if F32(1.0) == value || F64(1.0) == value => rhs,
            (_, Some(value)) if F32(1.0) == value || F64(1.0) == value => lhs,
            _ => return,
        };

        span_lint_and_sugg(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "ln(1 + x) can be computed more accurately",
            "consider using",
            format!("{}.ln_1p()", prepare_receiver_sugg(cx, recv)),
            Applicability::MachineApplicable,
        );
    }
}

// Returns an integer if the float constant is a whole number and it can be
// converted to an integer without loss of precision. For now we only check
// ranges [-16777215, 16777216) for type f32 as whole number floats outside
// this range are lossy and ambiguous.
#[expect(clippy::cast_possible_truncation)]
fn get_integer_from_float_constant(value: &Constant<'_>) -> Option<i32> {
    match value {
        F32(num) if num.fract() == 0.0 => {
            if (-16_777_215.0..16_777_216.0).contains(num) {
                Some(num.round() as i32)
            } else {
                None
            }
        },
        F64(num) if num.fract() == 0.0 => {
            if (-2_147_483_648.0..2_147_483_648.0).contains(num) {
                Some(num.round() as i32)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn check_powf(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    // Check receiver
    if let Some(value) = ConstEvalCtxt::new(cx).eval(receiver)
        && let Some(method) = if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            Some("exp")
        } else if F32(2.0) == value || F64(2.0) == value {
            Some("exp2")
        } else {
            None
        }
    {
        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "exponent for bases 2 and e can be computed more accurately",
            "consider using",
            format!("{}.{method}()", prepare_receiver_sugg(cx, &args[0])),
            Applicability::MachineApplicable,
        );
    }

    // Check argument
    if let Some(value) = ConstEvalCtxt::new(cx).eval(&args[0]) {
        let (lint, help, suggestion) = if F32(1.0 / 2.0) == value || F64(1.0 / 2.0) == value {
            (
                SUBOPTIMAL_FLOPS,
                "square-root of a number can be computed more efficiently and accurately",
                format!("{}.sqrt()", Sugg::hir(cx, receiver, "..").maybe_paren()),
            )
        } else if F32(1.0 / 3.0) == value || F64(1.0 / 3.0) == value {
            (
                IMPRECISE_FLOPS,
                "cube-root of a number can be computed more accurately",
                format!("{}.cbrt()", Sugg::hir(cx, receiver, "..").maybe_paren()),
            )
        } else if let Some(exponent) = get_integer_from_float_constant(&value) {
            (
                SUBOPTIMAL_FLOPS,
                "exponentiation with integer powers can be computed more efficiently",
                format!(
                    "{}.powi({})",
                    Sugg::hir(cx, receiver, "..").maybe_paren(),
                    numeric_literal::format(&exponent.to_string(), None, false)
                ),
            )
        } else {
            return;
        };

        span_lint_and_sugg(
            cx,
            lint,
            expr.span,
            help,
            "consider using",
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

fn check_powi(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    if let Some(value) = ConstEvalCtxt::new(cx).eval(&args[0])
        && value == Int(2)
        && let Some(parent) = get_parent_expr(cx, expr)
    {
        if let Some(grandparent) = get_parent_expr(cx, parent)
            && let ExprKind::MethodCall(PathSegment { ident: method, .. }, receiver, ..) = grandparent.kind
            && method.name == sym::sqrt
            && detect_hypot(cx, receiver).is_some()
        {
            return;
        }

        if let ExprKind::Binary(
            Spanned {
                node: op @ (BinOpKind::Add | BinOpKind::Sub),
                ..
            },
            lhs,
            rhs,
        ) = parent.kind
        {
            let other_addend = if lhs.hir_id == expr.hir_id { rhs } else { lhs };

            // Negate expr if original code has subtraction and expr is on the right side
            let maybe_neg_sugg = |expr, hir_id| {
                let sugg = Sugg::hir(cx, expr, "..");
                if matches!(op, BinOpKind::Sub) && hir_id == rhs.hir_id {
                    -sugg
                } else {
                    sugg
                }
            };

            span_lint_and_sugg(
                cx,
                SUBOPTIMAL_FLOPS,
                parent.span,
                "multiply and add expressions can be calculated more efficiently and accurately",
                "consider using",
                format!(
                    "{}.mul_add({}, {})",
                    Sugg::hir(cx, receiver, "..").maybe_paren(),
                    maybe_neg_sugg(receiver, expr.hir_id),
                    maybe_neg_sugg(other_addend, other_addend.hir_id),
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn detect_hypot(cx: &LateContext<'_>, receiver: &Expr<'_>) -> Option<String> {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Add, ..
        },
        add_lhs,
        add_rhs,
    ) = receiver.kind
    {
        // check if expression of the form x * x + y * y
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Mul, ..
            },
            lmul_lhs,
            lmul_rhs,
        ) = add_lhs.kind
            && let ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Mul, ..
                },
                rmul_lhs,
                rmul_rhs,
            ) = add_rhs.kind
            && eq_expr_value(cx, lmul_lhs, lmul_rhs)
            && eq_expr_value(cx, rmul_lhs, rmul_rhs)
        {
            return Some(format!(
                "{}.hypot({})",
                Sugg::hir(cx, lmul_lhs, "..").maybe_paren(),
                Sugg::hir(cx, rmul_lhs, "..")
            ));
        }

        // check if expression of the form x.powi(2) + y.powi(2)
        if let ExprKind::MethodCall(PathSegment { ident: lmethod, .. }, largs_0, [largs_1, ..], _) = &add_lhs.kind
            && let ExprKind::MethodCall(PathSegment { ident: rmethod, .. }, rargs_0, [rargs_1, ..], _) = &add_rhs.kind
            && lmethod.name == sym::powi
            && rmethod.name == sym::powi
            && let ecx = ConstEvalCtxt::new(cx)
            && let Some(lvalue) = ecx.eval(largs_1)
            && let Some(rvalue) = ecx.eval(rargs_1)
            && Int(2) == lvalue
            && Int(2) == rvalue
        {
            return Some(format!(
                "{}.hypot({})",
                Sugg::hir(cx, largs_0, "..").maybe_paren(),
                Sugg::hir(cx, rargs_0, "..")
            ));
        }
    }

    None
}

fn check_hypot(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    if let Some(message) = detect_hypot(cx, receiver) {
        span_lint_and_sugg(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "hypotenuse can be computed more accurately",
            "consider using",
            message,
            Applicability::MachineApplicable,
        );
    }
}

// TODO: Lint expressions of the form `x.exp() - y` where y > 1
// and suggest usage of `x.exp_m1() - (y - 1)` instead
fn check_expm1(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Sub, ..
        },
        lhs,
        rhs,
    ) = expr.kind
        && let ExprKind::MethodCall(path, self_arg, [], _) = &lhs.kind
        && path.ident.name == sym::exp
        && cx.typeck_results().expr_ty(lhs).is_floating_point()
        && let Some(value) = ConstEvalCtxt::new(cx).eval(rhs)
        && (F32(1.0) == value || F64(1.0) == value)
        && cx.typeck_results().expr_ty(self_arg).is_floating_point()
    {
        span_lint_and_sugg(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "(e.pow(x) - 1) can be computed more accurately",
            "consider using",
            format!("{}.exp_m1()", Sugg::hir(cx, self_arg, "..").maybe_paren()),
            Applicability::MachineApplicable,
        );
    }
}

fn is_float_mul_expr<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<(&'a Expr<'a>, &'a Expr<'a>)> {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Mul, ..
        },
        lhs,
        rhs,
    ) = &expr.kind
        && cx.typeck_results().expr_ty(lhs).is_floating_point()
        && cx.typeck_results().expr_ty(rhs).is_floating_point()
    {
        return Some((lhs, rhs));
    }

    None
}

// TODO: Fix rust-lang/rust-clippy#4735
fn check_mul_add(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: op @ (BinOpKind::Add | BinOpKind::Sub),
            ..
        },
        lhs,
        rhs,
    ) = &expr.kind
    {
        if let Some(parent) = get_parent_expr(cx, expr)
            && let ExprKind::MethodCall(PathSegment { ident: method, .. }, receiver, ..) = parent.kind
            && method.name == sym::sqrt
            && detect_hypot(cx, receiver).is_some()
        {
            return;
        }

        let maybe_neg_sugg = |expr| {
            let sugg = Sugg::hir(cx, expr, "..");
            if let BinOpKind::Sub = op { -sugg } else { sugg }
        };

        let (recv, arg1, arg2) = if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, lhs)
            && cx.typeck_results().expr_ty(rhs).is_floating_point()
        {
            (inner_lhs, Sugg::hir(cx, inner_rhs, ".."), maybe_neg_sugg(rhs))
        } else if let Some((inner_lhs, inner_rhs)) = is_float_mul_expr(cx, rhs)
            && cx.typeck_results().expr_ty(lhs).is_floating_point()
        {
            (inner_lhs, maybe_neg_sugg(inner_rhs), Sugg::hir(cx, lhs, ".."))
        } else {
            return;
        };

        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "multiply and add expressions can be calculated more efficiently and accurately",
            "consider using",
            format!("{}.mul_add({arg1}, {arg2})", prepare_receiver_sugg(cx, recv)),
            Applicability::MachineApplicable,
        );
    }
}

/// Returns true iff expr is an expression which tests whether or not
/// test is positive or an expression which tests whether or not test
/// is nonnegative.
/// Used for check-custom-abs function below
fn is_testing_positive(cx: &LateContext<'_>, expr: &Expr<'_>, test: &Expr<'_>) -> bool {
    if let ExprKind::Binary(Spanned { node: op, .. }, left, right) = expr.kind {
        match op {
            BinOpKind::Gt | BinOpKind::Ge => is_zero(cx, right) && eq_expr_value(cx, left, test),
            BinOpKind::Lt | BinOpKind::Le => is_zero(cx, left) && eq_expr_value(cx, right, test),
            _ => false,
        }
    } else {
        false
    }
}

/// See [`is_testing_positive`]
fn is_testing_negative(cx: &LateContext<'_>, expr: &Expr<'_>, test: &Expr<'_>) -> bool {
    if let ExprKind::Binary(Spanned { node: op, .. }, left, right) = expr.kind {
        match op {
            BinOpKind::Gt | BinOpKind::Ge => is_zero(cx, left) && eq_expr_value(cx, right, test),
            BinOpKind::Lt | BinOpKind::Le => is_zero(cx, right) && eq_expr_value(cx, left, test),
            _ => false,
        }
    } else {
        false
    }
}

/// Returns true iff expr is some zero literal
fn is_zero(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match ConstEvalCtxt::new(cx).eval_simple(expr) {
        Some(Int(i)) => i == 0,
        Some(F32(f)) => f == 0.0,
        Some(F64(f)) => f == 0.0,
        _ => false,
    }
}

/// If the two expressions are negations of each other, then it returns
/// a tuple, in which the first element is true iff expr1 is the
/// positive expressions, and the second element is the positive
/// one of the two expressions
/// If the two expressions are not negations of each other, then it
/// returns None.
fn are_negated<'a>(cx: &LateContext<'_>, expr1: &'a Expr<'a>, expr2: &'a Expr<'a>) -> Option<(bool, &'a Expr<'a>)> {
    if let ExprKind::Unary(UnOp::Neg, expr1_negated) = &expr1.kind
        && eq_expr_value(cx, expr1_negated, expr2)
    {
        return Some((false, expr2));
    }
    if let ExprKind::Unary(UnOp::Neg, expr2_negated) = &expr2.kind
        && eq_expr_value(cx, expr1, expr2_negated)
    {
        return Some((true, expr1));
    }
    None
}

fn check_custom_abs(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let Some(higher::If {
        cond,
        then,
        r#else: Some(r#else),
    }) = higher::If::hir(expr)
        && let if_body_expr = peel_blocks(then)
        && let else_body_expr = peel_blocks(r#else)
        && let Some((if_expr_positive, body)) = are_negated(cx, if_body_expr, else_body_expr)
    {
        let positive_abs_sugg = (
            "manual implementation of `abs` method",
            format!("{}.abs()", Sugg::hir(cx, body, "..").maybe_paren()),
        );
        let negative_abs_sugg = (
            "manual implementation of negation of `abs` method",
            format!("-{}.abs()", Sugg::hir(cx, body, "..").maybe_paren()),
        );
        let sugg = if is_testing_positive(cx, cond, body) {
            if if_expr_positive {
                positive_abs_sugg
            } else {
                negative_abs_sugg
            }
        } else if is_testing_negative(cx, cond, body) {
            if if_expr_positive {
                negative_abs_sugg
            } else {
                positive_abs_sugg
            }
        } else {
            return;
        };
        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            sugg.0,
            "try",
            sugg.1,
            Applicability::MachineApplicable,
        );
    }
}

fn are_same_base_logs(cx: &LateContext<'_>, expr_a: &Expr<'_>, expr_b: &Expr<'_>) -> bool {
    if let ExprKind::MethodCall(PathSegment { ident: method_a, .. }, _, args_a, _) = expr_a.kind
        && let ExprKind::MethodCall(PathSegment { ident: method_b, .. }, _, args_b, _) = expr_b.kind
    {
        return method_a.name == method_b.name
            && args_a.len() == args_b.len()
            && (matches!(method_a.name, sym::ln | sym::log2 | sym::log10)
                || method_a.name == sym::log && args_a.len() == 1 && eq_expr_value(cx, &args_a[0], &args_b[0]));
    }

    false
}

fn check_log_division(cx: &LateContext<'_>, expr: &Expr<'_>) {
    // check if expression of the form x.logN() / y.logN()
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Div, ..
        },
        lhs,
        rhs,
    ) = &expr.kind
        && are_same_base_logs(cx, lhs, rhs)
        && let ExprKind::MethodCall(_, largs_self, ..) = &lhs.kind
        && let ExprKind::MethodCall(_, rargs_self, ..) = &rhs.kind
    {
        span_lint_and_sugg(
            cx,
            SUBOPTIMAL_FLOPS,
            expr.span,
            "log base can be expressed more clearly",
            "consider using",
            format!(
                "{}.log({})",
                Sugg::hir(cx, largs_self, "..").maybe_paren(),
                Sugg::hir(cx, rargs_self, ".."),
            ),
            Applicability::MachineApplicable,
        );
    }
}

fn check_radians(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Div, ..
        },
        div_lhs,
        div_rhs,
    ) = &expr.kind
        && let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Mul, ..
            },
            mul_lhs,
            mul_rhs,
        ) = &div_lhs.kind
        && let ecx = ConstEvalCtxt::new(cx)
        && let Some(rvalue) = ecx.eval(div_rhs)
        && let Some(lvalue) = ecx.eval(mul_rhs)
    {
        // TODO: also check for constant values near PI/180 or 180/PI
        if (F32(f32_consts::PI) == rvalue || F64(f64_consts::PI) == rvalue)
            && (F32(180_f32) == lvalue || F64(180_f64) == lvalue)
        {
            let mut proposal = format!("{}.to_degrees()", Sugg::hir(cx, mul_lhs, "..").maybe_paren());
            if let ExprKind::Lit(literal) = mul_lhs.kind
                && let ast::LitKind::Float(ref value, float_type) = literal.node
                && float_type == ast::LitFloatType::Unsuffixed
            {
                if value.as_str().ends_with('.') {
                    proposal = format!("{}0_f64.to_degrees()", Sugg::hir(cx, mul_lhs, ".."));
                } else {
                    proposal = format!("{}_f64.to_degrees()", Sugg::hir(cx, mul_lhs, ".."));
                }
            }
            span_lint_and_sugg(
                cx,
                SUBOPTIMAL_FLOPS,
                expr.span,
                "conversion to degrees can be done more accurately",
                "consider using",
                proposal,
                Applicability::MachineApplicable,
            );
        } else if (F32(180_f32) == rvalue || F64(180_f64) == rvalue)
            && (F32(f32_consts::PI) == lvalue || F64(f64_consts::PI) == lvalue)
        {
            let mut proposal = format!("{}.to_radians()", Sugg::hir(cx, mul_lhs, "..").maybe_paren());
            if let ExprKind::Lit(literal) = mul_lhs.kind
                && let ast::LitKind::Float(ref value, float_type) = literal.node
                && float_type == ast::LitFloatType::Unsuffixed
            {
                if value.as_str().ends_with('.') {
                    proposal = format!("{}0_f64.to_radians()", Sugg::hir(cx, mul_lhs, ".."));
                } else {
                    proposal = format!("{}_f64.to_radians()", Sugg::hir(cx, mul_lhs, ".."));
                }
            }
            span_lint_and_sugg(
                cx,
                SUBOPTIMAL_FLOPS,
                expr.span,
                "conversion to radians can be done more accurately",
                "consider using",
                proposal,
                Applicability::MachineApplicable,
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for FloatingPointArithmetic {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // All of these operations are currently not const and are in std.
        if is_in_const_context(cx) {
            return;
        }

        if let ExprKind::MethodCall(path, receiver, args, _) = &expr.kind {
            let recv_ty = cx.typeck_results().expr_ty(receiver);

            if recv_ty.is_floating_point() && !is_no_std_crate(cx) && is_inherent_method_call(cx, expr) {
                match path.ident.name {
                    sym::ln => check_ln1p(cx, expr, receiver),
                    sym::log => check_log_base(cx, expr, receiver, args),
                    sym::powf => check_powf(cx, expr, receiver, args),
                    sym::powi => check_powi(cx, expr, receiver, args),
                    sym::sqrt => check_hypot(cx, expr, receiver),
                    _ => {},
                }
            }
        } else {
            if !is_no_std_crate(cx) {
                check_expm1(cx, expr);
                check_mul_add(cx, expr);
                check_custom_abs(cx, expr);
                check_log_division(cx, expr);
            }
            check_radians(cx, expr);
        }
    }
}
