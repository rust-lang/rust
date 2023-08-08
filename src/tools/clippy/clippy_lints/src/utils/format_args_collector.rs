use clippy_utils::macros::collect_ast_format_args;
use clippy_utils::source::snippet_opt;
use itertools::Itertools;
use rustc_ast::{Expr, ExprKind, FormatArgs};
use rustc_lexer::{tokenize, TokenKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene;
use std::iter::once;

declare_clippy_lint! {
    /// ### What it does
    /// Collects [`rustc_ast::FormatArgs`] so that future late passes can call
    /// [`clippy_utils::macros::find_format_args`]
    pub FORMAT_ARGS_COLLECTOR,
    internal_warn,
    "collects `format_args` AST nodes for use in later lints"
}

declare_lint_pass!(FormatArgsCollector => [FORMAT_ARGS_COLLECTOR]);

impl EarlyLintPass for FormatArgsCollector {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::FormatArgs(args) = &expr.kind {
            if has_span_from_proc_macro(cx, args) {
                return;
            }

            collect_ast_format_args(expr.span, args);
        }
    }
}

/// Detects if the format string or an argument has its span set by a proc macro to something inside
/// a macro callsite, e.g.
///
/// ```ignore
/// println!(some_proc_macro!("input {}"), a);
/// ```
///
/// Where `some_proc_macro` expands to
///
/// ```ignore
/// println!("output {}", a);
/// ```
///
/// But with the span of `"output {}"` set to the macro input
///
/// ```ignore
/// println!(some_proc_macro!("input {}"), a);
/// //                        ^^^^^^^^^^
/// ```
fn has_span_from_proc_macro(cx: &EarlyContext<'_>, args: &FormatArgs) -> bool {
    let ctxt = args.span.ctxt();

    // `format!("{} {} {c}", "one", "two", c = "three")`
    //                       ^^^^^  ^^^^^      ^^^^^^^
    let argument_span = args
        .arguments
        .explicit_args()
        .iter()
        .map(|argument| hygiene::walk_chain(argument.expr.span, ctxt));

    // `format!("{} {} {c}", "one", "two", c = "three")`
    //                     ^^     ^^     ^^^^^^
    let between_spans = once(args.span)
        .chain(argument_span)
        .tuple_windows()
        .map(|(start, end)| start.between(end));

    for between_span in between_spans {
        let mut seen_comma = false;

        let Some(snippet) = snippet_opt(cx, between_span) else {
            return true;
        };
        for token in tokenize(&snippet) {
            match token.kind {
                TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace => {},
                TokenKind::Comma if !seen_comma => seen_comma = true,
                // named arguments, `start_val, name = end_val`
                //                            ^^^^^^^^^ between_span
                TokenKind::Ident | TokenKind::Eq if seen_comma => {},
                // An unexpected token usually indicates that we crossed a macro boundary
                //
                // `println!(some_proc_macro!("input {}"), a)`
                //                                      ^^^ between_span
                // `println!("{}", val!(x))`
                //               ^^^^^^^ between_span
                _ => return true,
            }
        }

        if !seen_comma {
            return true;
        }
    }

    false
}
