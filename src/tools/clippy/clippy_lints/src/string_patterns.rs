use std::ops::ControlFlow;

use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::macros::matching_root_macro_call;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{snippet, str_literal_to_char_literal};
use clippy_utils::visitors::{Descend, for_each_expr};
use clippy_utils::{path_to_local_id, sym};
use itertools::Itertools;
use rustc_ast::{BinOpKind, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PatExprKind, PatKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual `char` comparison in string patterns
    ///
    /// ### Why is this bad?
    /// This can be written more concisely using a `char` or an array of `char`.
    /// This is more readable and more optimized when comparing to only one `char`.
    ///
    /// ### Example
    /// ```no_run
    /// "Hello World!".trim_end_matches(|c| c == '.' || c == ',' || c == '!' || c == '?');
    /// ```
    /// Use instead:
    /// ```no_run
    /// "Hello World!".trim_end_matches(['.', ',', '!', '?']);
    /// ```
    #[clippy::version = "1.81.0"]
    pub MANUAL_PATTERN_CHAR_COMPARISON,
    style,
    "manual char comparison in string patterns"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for string methods that receive a single-character
    /// `str` as an argument, e.g., `_.split("x")`.
    ///
    /// ### Why is this bad?
    /// While this can make a perf difference on some systems,
    /// benchmarks have proven inconclusive. But at least using a
    /// char literal makes it clear that we are looking at a single
    /// character.
    ///
    /// ### Known problems
    /// Does not catch multi-byte unicode characters. This is by
    /// design, on many machines, splitting by a non-ascii char is
    /// actually slower. Please do your own measurements instead of
    /// relying solely on the results of this lint.
    ///
    /// ### Example
    /// ```rust,ignore
    /// _.split("x");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// _.split('x');
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SINGLE_CHAR_PATTERN,
    pedantic,
    "using a single-character str where a char could be used, e.g., `_.split(\"x\")`"
}

pub struct StringPatterns {
    msrv: Msrv,
}

impl StringPatterns {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(StringPatterns => [MANUAL_PATTERN_CHAR_COMPARISON, SINGLE_CHAR_PATTERN]);

const PATTERN_METHODS: [(Symbol, usize); 22] = [
    (sym::contains, 0),
    (sym::starts_with, 0),
    (sym::ends_with, 0),
    (sym::find, 0),
    (sym::rfind, 0),
    (sym::split, 0),
    (sym::split_inclusive, 0),
    (sym::rsplit, 0),
    (sym::split_terminator, 0),
    (sym::rsplit_terminator, 0),
    (sym::splitn, 1),
    (sym::rsplitn, 1),
    (sym::split_once, 0),
    (sym::rsplit_once, 0),
    (sym::matches, 0),
    (sym::rmatches, 0),
    (sym::match_indices, 0),
    (sym::rmatch_indices, 0),
    (sym::trim_start_matches, 0),
    (sym::trim_end_matches, 0),
    (sym::replace, 0),
    (sym::replacen, 0),
];

fn check_single_char_pattern_lint(cx: &LateContext<'_>, arg: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(hint) = str_literal_to_char_literal(cx, arg, &mut applicability, true) {
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_PATTERN,
            arg.span,
            "single-character string constant used as pattern",
            "consider using a `char`",
            hint,
            applicability,
        );
    }
}

fn get_char_span<'tcx>(cx: &'_ LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<Span> {
    if cx.typeck_results().expr_ty_adjusted(expr).is_char()
        && !expr.span.from_expansion()
        && switch_to_eager_eval(cx, expr)
    {
        Some(expr.span)
    } else {
        None
    }
}

fn check_manual_pattern_char_comparison(cx: &LateContext<'_>, method_arg: &Expr<'_>, msrv: Msrv) {
    if let ExprKind::Closure(closure) = method_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let Some(PatKind::Binding(_, binding, ..)) = body.params.first().map(|p| p.pat.kind)
    {
        let mut set_char_spans: Vec<Span> = Vec::new();

        // We want to retrieve all the comparisons done.
        // They are ordered in a nested way and so we need to traverse the AST to collect them all.
        if for_each_expr(cx, body.value, |sub_expr| -> ControlFlow<(), Descend> {
            match sub_expr.kind {
                ExprKind::Binary(op, left, right) if op.node == BinOpKind::Eq => {
                    if path_to_local_id(left, binding)
                        && let Some(span) = get_char_span(cx, right)
                    {
                        set_char_spans.push(span);
                        ControlFlow::Continue(Descend::No)
                    } else if path_to_local_id(right, binding)
                        && let Some(span) = get_char_span(cx, left)
                    {
                        set_char_spans.push(span);
                        ControlFlow::Continue(Descend::No)
                    } else {
                        ControlFlow::Break(())
                    }
                },
                ExprKind::Binary(op, _, _) if op.node == BinOpKind::Or => ControlFlow::Continue(Descend::Yes),
                ExprKind::Match(match_value, [arm, _], _) => {
                    if matching_root_macro_call(cx, sub_expr.span, sym::matches_macro).is_none()
                        || arm.guard.is_some()
                        || !path_to_local_id(match_value, binding)
                    {
                        return ControlFlow::Break(());
                    }
                    if arm.pat.walk_short(|pat| match pat.kind {
                        PatKind::Expr(expr) if let PatExprKind::Lit { lit, negated: false } = expr.kind => {
                            if let LitKind::Char(_) = lit.node {
                                set_char_spans.push(lit.span);
                            }
                            true
                        },
                        PatKind::Or(_) => true,
                        _ => false,
                    }) {
                        ControlFlow::Continue(Descend::No)
                    } else {
                        ControlFlow::Break(())
                    }
                },
                _ => ControlFlow::Break(()),
            }
        })
        .is_some()
        {
            return;
        }
        if set_char_spans.len() > 1 && !msrv.meets(cx, msrvs::PATTERN_TRAIT_CHAR_ARRAY) {
            return;
        }
        span_lint_and_then(
            cx,
            MANUAL_PATTERN_CHAR_COMPARISON,
            method_arg.span,
            "this manual char comparison can be written more succinctly",
            |diag| {
                if let [set_char_span] = set_char_spans[..] {
                    diag.span_suggestion(
                        method_arg.span,
                        "consider using a `char`",
                        snippet(cx, set_char_span, "c"),
                        Applicability::MachineApplicable,
                    );
                } else {
                    diag.span_suggestion(
                        method_arg.span,
                        "consider using an array of `char`",
                        format!(
                            "[{}]",
                            set_char_spans.into_iter().map(|span| snippet(cx, span, "c")).join(", ")
                        ),
                        Applicability::MachineApplicable,
                    );
                }
            },
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for StringPatterns {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let ExprKind::MethodCall(method, receiver, args, _) = expr.kind
            && let ty::Ref(_, ty, _) = cx.typeck_results().expr_ty_adjusted(receiver).kind()
            && ty.is_str()
            && let method_name = method.ident.name
            && let Some(&(_, pos)) = PATTERN_METHODS
                .iter()
                .find(|(array_method_name, _)| *array_method_name == method_name)
            && let Some(arg) = args.get(pos)
        {
            check_single_char_pattern_lint(cx, arg);

            check_manual_pattern_char_comparison(cx, arg, self.msrv);
        }
    }
}
