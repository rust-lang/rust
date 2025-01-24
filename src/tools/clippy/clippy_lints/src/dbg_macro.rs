use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_test;
use clippy_utils::macros::{MacroCall, macro_backtrace};
use clippy_utils::source::snippet_with_applicability;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, SyntaxContext, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the [`dbg!`](https://doc.rust-lang.org/std/macro.dbg.html) macro.
    ///
    /// ### Why restrict this?
    /// The `dbg!` macro is intended as a debugging tool. It should not be present in released
    /// software or committed to a version control system.
    ///
    /// ### Example
    /// ```rust,ignore
    /// dbg!(true)
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// true
    /// ```
    #[clippy::version = "1.34.0"]
    pub DBG_MACRO,
    restriction,
    "`dbg!` macro is intended as a debugging tool"
}

pub struct DbgMacro {
    allow_dbg_in_tests: bool,
    /// Tracks the `dbg!` macro callsites that are already checked.
    checked_dbg_call_site: FxHashSet<Span>,
    /// Tracks the previous `SyntaxContext`, to avoid walking the same context chain.
    prev_ctxt: SyntaxContext,
}

impl_lint_pass!(DbgMacro => [DBG_MACRO]);

impl DbgMacro {
    pub fn new(conf: &'static Conf) -> Self {
        DbgMacro {
            allow_dbg_in_tests: conf.allow_dbg_in_tests,
            checked_dbg_call_site: FxHashSet::default(),
            prev_ctxt: SyntaxContext::root(),
        }
    }
}

impl LateLintPass<'_> for DbgMacro {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let cur_syntax_ctxt = expr.span.ctxt();

        if cur_syntax_ctxt != self.prev_ctxt &&
            let Some(macro_call) = first_dbg_macro_in_expansion(cx, expr.span) &&
            !in_external_macro(cx.sess(), macro_call.span) &&
            self.checked_dbg_call_site.insert(macro_call.span) &&
            // allows `dbg!` in test code if allow-dbg-in-test is set to true in clippy.toml
            !(self.allow_dbg_in_tests && is_in_test(cx.tcx, expr.hir_id))
        {
            self.prev_ctxt = cur_syntax_ctxt;

            span_lint_and_then(
                cx,
                DBG_MACRO,
                macro_call.span,
                "the `dbg!` macro is intended as a debugging tool",
                |diag| {
                    let mut applicability = Applicability::MachineApplicable;

                    let (sugg_span, suggestion) = match expr.peel_drop_temps().kind {
                        // dbg!()
                        ExprKind::Block(..) => {
                            // If the `dbg!` macro is a "free" statement and not contained within other expressions,
                            // remove the whole statement.
                            if let Node::Stmt(_) = cx.tcx.parent_hir_node(expr.hir_id)
                                && let Some(semi_span) = cx.sess().source_map().mac_call_stmt_semi_span(macro_call.span)
                            {
                                (macro_call.span.to(semi_span), String::new())
                            } else {
                                (macro_call.span, String::from("()"))
                            }
                        },
                        // dbg!(1)
                        ExprKind::Match(val, ..) => (
                            macro_call.span,
                            snippet_with_applicability(cx, val.span.source_callsite(), "..", &mut applicability)
                                .to_string(),
                        ),
                        // dbg!(2, 3)
                        ExprKind::Tup(
                            [
                                Expr {
                                    kind: ExprKind::Match(first, ..),
                                    ..
                                },
                                ..,
                                Expr {
                                    kind: ExprKind::Match(last, ..),
                                    ..
                                },
                            ],
                        ) => {
                            let snippet = snippet_with_applicability(
                                cx,
                                first.span.source_callsite().to(last.span.source_callsite()),
                                "..",
                                &mut applicability,
                            );
                            (macro_call.span, format!("({snippet})"))
                        },
                        _ => unreachable!(),
                    };

                    diag.span_suggestion(
                        sugg_span,
                        "remove the invocation before committing it to a version control system",
                        suggestion,
                        applicability,
                    );
                },
            );
        }
    }

    fn check_crate_post(&mut self, _: &LateContext<'_>) {
        self.checked_dbg_call_site = FxHashSet::default();
    }
}

fn first_dbg_macro_in_expansion(cx: &LateContext<'_>, span: Span) -> Option<MacroCall> {
    macro_backtrace(span).find(|mc| cx.tcx.is_diagnostic_item(sym::dbg_macro, mc.def_id))
}
