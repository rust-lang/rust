use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{FormatArgsStorage, format_args_inputs_span, root_macro_call_first_node};
use clippy_utils::res::MaybeDef;
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::std_or_core;
use rustc_errors::Applicability;
use rustc_hir::{AssignOpKind, Expr, ExprKind, LangItem, MatchSource};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Detects cases where the result of a `format!` call is
    /// appended to an existing `String`.
    ///
    /// ### Why is this bad?
    /// Introduces an extra, avoidable heap allocation.
    ///
    /// ### Known problems
    /// `format!` returns a `String` but `write!` returns a `Result`.
    /// Thus you are forced to ignore the `Err` variant to achieve the same API.
    ///
    /// While using `write!` in the suggested way should never fail, this isn't necessarily clear to the programmer.
    ///
    /// ### Example
    /// ```no_run
    /// let mut s = String::new();
    /// s += &format!("0x{:X}", 1024);
    /// s.push_str(&format!("0x{:X}", 1024));
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::fmt::Write as _; // import without risk of name clashing
    ///
    /// let mut s = String::new();
    /// let _ = write!(s, "0x{:X}", 1024);
    /// ```
    #[clippy::version = "1.62.0"]
    pub FORMAT_PUSH_STRING,
    pedantic,
    "`format!(..)` appended to existing `String`"
}
impl_lint_pass!(FormatPushString => [FORMAT_PUSH_STRING]);

pub(crate) struct FormatPushString {
    format_args: FormatArgsStorage,
}

enum FormatSearchResults {
    /// The expression is itself a `format!()` invocation -- we can make a suggestion to replace it
    Direct(Span),
    /// The expression contains zero or more `format!()`s, e.g.:
    /// ```ignore
    /// if true {
    ///     format!("hello")
    /// } else {
    ///     format!("world")
    /// }
    /// ```
    /// or
    /// ```ignore
    /// match true {
    ///     true => format!("hello"),
    ///     false => format!("world"),
    /// }
    Nested(Vec<Span>),
}

impl FormatPushString {
    pub(crate) fn new(format_args: FormatArgsStorage) -> Self {
        Self { format_args }
    }

    fn find_formats<'tcx>(&self, cx: &LateContext<'_>, e: &'tcx Expr<'tcx>) -> FormatSearchResults {
        let expr_as_format = |e| {
            if let Some(macro_call) = root_macro_call_first_node(cx, e)
                && cx.tcx.is_diagnostic_item(sym::format_macro, macro_call.def_id)
                && let Some(format_args) = self.format_args.get(cx, e, macro_call.expn)
            {
                Some(format_args_inputs_span(format_args))
            } else {
                None
            }
        };

        let e = e.peel_blocks().peel_borrows();
        if let Some(fmt) = expr_as_format(e) {
            FormatSearchResults::Direct(fmt)
        } else {
            fn inner<'tcx>(
                e: &'tcx Expr<'tcx>,
                expr_as_format: &impl Fn(&'tcx Expr<'tcx>) -> Option<Span>,
                out: &mut Vec<Span>,
            ) {
                let e = e.peel_blocks().peel_borrows();

                match e.kind {
                    _ if expr_as_format(e).is_some() => out.push(e.span),
                    ExprKind::Match(_, arms, MatchSource::Normal) => {
                        for arm in arms {
                            inner(arm.body, expr_as_format, out);
                        }
                    },
                    ExprKind::If(_, then, els) => {
                        inner(then, expr_as_format, out);
                        if let Some(els) = els {
                            inner(els, expr_as_format, out);
                        }
                    },
                    _ => {},
                }
            }
            let mut spans = vec![];
            inner(e, &expr_as_format, &mut spans);
            FormatSearchResults::Nested(spans)
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for FormatPushString {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (recv, arg) = match expr.kind {
            ExprKind::MethodCall(_, recv, [arg], _) => {
                if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && cx.tcx.is_diagnostic_item(sym::string_push_str, fn_def_id)
                {
                    (recv, arg)
                } else {
                    return;
                }
            },
            ExprKind::AssignOp(op, recv, arg) if op.node == AssignOpKind::AddAssign && is_string(cx, recv) => {
                (recv, arg)
            },
            _ => return,
        };
        let Some(std_or_core) = std_or_core(cx) else {
            // not even `core` is available, so can't suggest `write!`
            return;
        };
        match self.find_formats(cx, arg) {
            FormatSearchResults::Direct(format_args) => {
                span_lint_and_then(
                    cx,
                    FORMAT_PUSH_STRING,
                    expr.span,
                    "`format!(..)` appended to existing `String`",
                    |diag| {
                        let mut app = Applicability::MaybeIncorrect;
                        let msg = "consider using `write!` to avoid the extra allocation";

                        let sugg = format!(
                            "let _ = write!({recv}, {format_args})",
                            recv = snippet_with_context(cx.sess(), recv.span, expr.span.ctxt(), "_", &mut app).0,
                            format_args = snippet_with_applicability(cx.sess(), format_args, "..", &mut app),
                        );
                        diag.span_suggestion_verbose(expr.span, msg, sugg, app);

                        // TODO: omit the note if the `Write` trait is imported at point
                        // Tip: `TyCtxt::in_scope_traits` isn't it -- it returns a non-empty list only when called on
                        // the `HirId` of a `ExprKind::MethodCall` that is a call of a _trait_ method.
                        diag.note(format!("you may need to import the `{std_or_core}::fmt::Write` trait"));
                    },
                );
            },
            FormatSearchResults::Nested(spans) => {
                if !spans.is_empty() {
                    span_lint_and_then(
                        cx,
                        FORMAT_PUSH_STRING,
                        expr.span,
                        "`format!(..)` appended to existing `String`",
                        |diag| {
                            diag.help("consider using `write!` to avoid the extra allocation");
                            diag.span_labels(spans, "`format!` used here");

                            // TODO: omit the note if the `Write` trait is imported at point
                            // Tip: `TyCtxt::in_scope_traits` isn't it -- it returns a non-empty list only when called
                            // on the `HirId` of a `ExprKind::MethodCall` that is a call of
                            // a _trait_ method.
                            diag.note(format!("you may need to import the `{std_or_core}::fmt::Write` trait"));
                        },
                    );
                }
            },
        }
    }
}

fn is_string(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    cx.typeck_results()
        .expr_ty(e)
        .peel_refs()
        .is_lang_item(cx, LangItem::String)
}
