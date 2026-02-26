use std::time::Duration;

use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint_and_note, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::sugg::Sugg;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::impl_lint_pass;
use rustc_span::SyntaxContext;

declare_clippy_lint! {
    /// ### What it does
    /// Lints subtraction between `Instant::now()` and another `Instant`.
    ///
    /// ### Why is this bad?
    /// It is easy to accidentally write `prev_instant - Instant::now()`, which will always be 0ns
    /// as `Instant` subtraction saturates.
    ///
    /// `prev_instant.elapsed()` also more clearly signals intention.
    ///
    /// ### Example
    /// ```no_run
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = Instant::now() - prev_instant;
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = prev_instant.elapsed();
    /// ```
    #[clippy::version = "1.65.0"]
    pub MANUAL_INSTANT_ELAPSED,
    pedantic,
    "subtraction between `Instant::now()` and previous `Instant`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints subtraction between an `Instant` and a `Duration`, or between two `Duration` values.
    ///
    /// ### Why is this bad?
    /// Unchecked subtraction could cause underflow on certain platforms, leading to
    /// unintentional panics.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now() - Duration::from_secs(5);
    /// let dur1 = Duration::from_secs(3);
    /// let dur2 = Duration::from_secs(5);
    /// let diff = dur1 - dur2;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now().checked_sub(Duration::from_secs(5));
    /// let dur1 = Duration::from_secs(3);
    /// let dur2 = Duration::from_secs(5);
    /// let diff = dur1.checked_sub(dur2);
    /// ```
    #[clippy::version = "1.67.0"]
    pub UNCHECKED_TIME_SUBTRACTION,
    pedantic,
    "finds unchecked subtraction involving 'Duration' or 'Instant'"
}

pub struct UncheckedTimeSubtraction {
    msrv: Msrv,
}

impl UncheckedTimeSubtraction {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(UncheckedTimeSubtraction => [MANUAL_INSTANT_ELAPSED, UNCHECKED_TIME_SUBTRACTION]);

impl LateLintPass<'_> for UncheckedTimeSubtraction {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        let (lhs, rhs) = match expr.kind {
            ExprKind::Binary(op, lhs, rhs) if matches!(op.node, BinOpKind::Sub,) => (lhs, rhs),
            ExprKind::MethodCall(_, lhs, [rhs], _) if cx.ty_based_def(expr).is_diag_item(cx, sym::sub) => (lhs, rhs),
            _ => return,
        };
        let typeck = cx.typeck_results();
        let lhs_name = typeck.expr_ty(lhs).opt_diag_name(cx);
        let rhs_name = typeck.expr_ty(rhs).opt_diag_name(cx);

        if lhs_name == Some(sym::Instant) {
            // Instant::now() - instant
            if is_instant_now_call(cx, lhs) && rhs_name == Some(sym::Instant) {
                print_manual_instant_elapsed_sugg(cx, expr, rhs);
            }
            // instant - duration
            else if rhs_name == Some(sym::Duration)
                && !expr.span.from_expansion()
                && self.msrv.meets(cx, msrvs::TRY_FROM)
            {
                print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr);
            }
        }
        // duration - duration
        else if lhs_name == Some(sym::Duration)
            && rhs_name == Some(sym::Duration)
            && !expr.span.from_expansion()
            && self.msrv.meets(cx, msrvs::TRY_FROM)
        {
            let const_eval = ConstEvalCtxt::new(cx);
            let ctxt = expr.span.ctxt();
            if let Some(lhs) = const_eval_duration(&const_eval, lhs, ctxt)
                && let Some(rhs) = const_eval_duration(&const_eval, rhs, ctxt)
            {
                if lhs >= rhs {
                    // If the duration subtraction can be proven to not underflow, then we don't lint
                    return;
                }

                span_lint_and_note(
                    cx,
                    UNCHECKED_TIME_SUBTRACTION,
                    expr.span,
                    "unchecked subtraction of two `Duration` that will underflow",
                    None,
                    "if this is intentional, consider allowing the lint",
                );
                return;
            }

            print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr);
        }
    }
}

fn is_instant_now_call(cx: &LateContext<'_>, expr_block: &'_ Expr<'_>) -> bool {
    if let ExprKind::Call(fn_expr, []) = expr_block.kind
        && cx.ty_based_def(fn_expr).is_diag_item(cx, sym::instant_now)
    {
        true
    } else {
        false
    }
}

/// Returns true if this subtraction is part of a chain like `(a - b) - c`
fn is_chained_time_subtraction(cx: &LateContext<'_>, lhs: &Expr<'_>) -> bool {
    if let ExprKind::Binary(op, inner_lhs, inner_rhs) = &lhs.kind
        && matches!(op.node, BinOpKind::Sub)
    {
        let typeck = cx.typeck_results();
        let left_ty = typeck.expr_ty(inner_lhs);
        let right_ty = typeck.expr_ty(inner_rhs);
        is_time_type(cx, left_ty) && is_time_type(cx, right_ty)
    } else {
        false
    }
}

/// Returns true if the type is Duration or Instant
fn is_time_type(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    matches!(ty.opt_diag_name(cx), Some(sym::Duration | sym::Instant))
}

fn print_manual_instant_elapsed_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, rhs: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    let sugg = Sugg::hir_with_context(cx, rhs, expr.span.ctxt(), "<instant>", &mut applicability);
    span_lint_and_sugg(
        cx,
        MANUAL_INSTANT_ELAPSED,
        expr.span,
        "manual implementation of `Instant::elapsed`",
        "try",
        format!("{}.elapsed()", sugg.maybe_paren()),
        applicability,
    );
}

fn print_unchecked_duration_subtraction_sugg(
    cx: &LateContext<'_>,
    left_expr: &Expr<'_>,
    right_expr: &Expr<'_>,
    expr: &Expr<'_>,
) {
    span_lint_and_then(
        cx,
        UNCHECKED_TIME_SUBTRACTION,
        expr.span,
        "unchecked subtraction of a `Duration`",
        |diag| {
            // For chained subtraction, like `(dur1 - dur2) - dur3` or `(instant - dur1) - dur2`,
            // avoid suggestions
            if !is_chained_time_subtraction(cx, left_expr) {
                let mut applicability = Applicability::MachineApplicable;
                let left_sugg = Sugg::hir_with_context(cx, left_expr, expr.span.ctxt(), "<left>", &mut applicability);
                let right_sugg =
                    Sugg::hir_with_context(cx, right_expr, expr.span.ctxt(), "<right>", &mut applicability);

                diag.span_suggestion(
                    expr.span,
                    "try",
                    format!("{}.checked_sub({}).unwrap()", left_sugg.maybe_paren(), right_sugg),
                    applicability,
                );
            }
        },
    );
}

fn const_eval_duration(const_eval: &ConstEvalCtxt<'_>, expr: &Expr<'_>, ctxt: SyntaxContext) -> Option<Duration> {
    if let ExprKind::Call(func, args) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(_, func_name)) = func.kind
    {
        macro_rules! try_parse_duration {
            (($( $name:ident : $var:ident ( $ty:ty ) ),+ $(,)?) -> $ctor:ident ( $($args:tt)* )) => {{
                let [$( $name ),+] = args else { return None };
                $(
                    let Some(Constant::$var(v)) = const_eval.eval_local($name, ctxt) else { return None };
                    let $name = <$ty>::try_from(v).ok()?;
                )+
                Some(Duration::$ctor($($args)*))
            }};
        }

        return match func_name.ident.name {
            sym::new => try_parse_duration! { (secs: Int(u64), nanos: Int(u32)) -> new(secs, nanos) },
            sym::from_nanos => try_parse_duration! { (nanos: Int(u64)) -> from_nanos(nanos) },
            sym::from_nanos_u128 => try_parse_duration! { (nanos: Int(u128)) -> from_nanos_u128(nanos) },
            sym::from_micros => try_parse_duration! { (micros: Int(u64)) -> from_micros(micros) },
            sym::from_millis => try_parse_duration! { (millis: Int(u64)) -> from_millis(millis) },
            sym::from_secs => try_parse_duration! { (secs: Int(u64)) -> from_secs(secs) },
            sym::from_secs_f32 => try_parse_duration! { (secs: F32(f32)) -> from_secs_f32(secs) },
            sym::from_secs_f64 => try_parse_duration! { (secs: F64(f64)) -> from_secs_f64(secs) },
            sym::from_mins => try_parse_duration! { (mins: Int(u64)) -> from_mins(mins) },
            sym::from_hours => {
                try_parse_duration! { (hours: Int(u64)) -> from_hours(hours) }
            },
            sym::from_days => {
                try_parse_duration! { (days: Int(u64)) -> from_hours(days * 24) }
            },
            sym::from_weeks => {
                try_parse_duration! { (weeks: Int(u64)) -> from_hours(weeks * 24 * 7) }
            },
            _ => None,
        };
    }

    None
}
