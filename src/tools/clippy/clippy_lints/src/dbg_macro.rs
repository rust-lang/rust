use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_in_cfg_test, is_in_test_function};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, BytePos, Pos, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the [`dbg!`](https://doc.rust-lang.org/std/macro.dbg.html) macro.
    ///
    /// ### Why is this bad?
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

/// Gets the span of the statement up to the next semicolon, if and only if the next
/// non-whitespace character actually is a semicolon.
/// E.g.
/// ```rust,ignore
/// 
///    dbg!();
///    ^^^^^^^  this span is returned
///
///   foo!(dbg!());
///             no span is returned
/// ```
fn span_including_semi(cx: &LateContext<'_>, span: Span) -> Option<Span> {
    let sm = cx.sess().source_map();
    let sf = sm.lookup_source_file(span.hi());
    let src = sf.src.as_ref()?.get(span.hi().to_usize()..)?;
    let first_non_whitespace = src.find(|c: char| !c.is_whitespace())?;

    if src.as_bytes()[first_non_whitespace] == b';' {
        let hi = span.hi() + BytePos::from_usize(first_non_whitespace + 1);
        Some(span.with_hi(hi))
    } else {
        None
    }
}

#[derive(Copy, Clone)]
pub struct DbgMacro {
    allow_dbg_in_tests: bool,
}

impl_lint_pass!(DbgMacro => [DBG_MACRO]);

impl DbgMacro {
    pub fn new(allow_dbg_in_tests: bool) -> Self {
        DbgMacro { allow_dbg_in_tests }
    }
}

impl LateLintPass<'_> for DbgMacro {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        if cx.tcx.is_diagnostic_item(sym::dbg_macro, macro_call.def_id) {
            // allows `dbg!` in test code if allow-dbg-in-test is set to true in clippy.toml
            if self.allow_dbg_in_tests
                && (is_in_test_function(cx.tcx, expr.hir_id) || is_in_cfg_test(cx.tcx, expr.hir_id))
            {
                return;
            }
            let mut applicability = Applicability::MachineApplicable;

            let (sugg_span, suggestion) = match expr.peel_drop_temps().kind {
                // dbg!()
                ExprKind::Block(..) => {
                    // If the `dbg!` macro is a "free" statement and not contained within other expressions,
                    // remove the whole statement.
                    if let Some(Node::Stmt(stmt)) = cx.tcx.hir().find_parent(expr.hir_id)
                        && let Some(span) = span_including_semi(cx, stmt.span.source_callsite())
                    {
                        (span, String::new())
                    } else {
                        (macro_call.span, String::from("()"))
                    }
                },
                // dbg!(1)
                ExprKind::Match(val, ..) => (
                    macro_call.span,
                    snippet_with_applicability(cx, val.span.source_callsite(), "..", &mut applicability).to_string(),
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
                _ => return,
            };

            span_lint_and_sugg(
                cx,
                DBG_MACRO,
                sugg_span,
                "the `dbg!` macro is intended as a debugging tool",
                "remove the invocation before committing it to a version control system",
                suggestion,
                applicability,
            );
        }
    }
}
