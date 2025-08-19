use clippy_utils::macros::FormatArgsStorage;
use clippy_utils::source::SpanRangeExt;
use itertools::Itertools;
use rustc_ast::{Crate, Expr, ExprKind, FormatArgs};
use rustc_data_structures::fx::FxHashMap;
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, hygiene};
use std::iter::once;
use std::mem;

/// Populates [`FormatArgsStorage`] with AST [`FormatArgs`] nodes
pub struct FormatArgsCollector {
    format_args: FxHashMap<Span, FormatArgs>,
    storage: FormatArgsStorage,
}

impl FormatArgsCollector {
    pub fn new(storage: FormatArgsStorage) -> Self {
        Self {
            format_args: FxHashMap::default(),
            storage,
        }
    }
}

impl_lint_pass!(FormatArgsCollector => []);

impl EarlyLintPass for FormatArgsCollector {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::FormatArgs(args) = &expr.kind {
            if has_span_from_proc_macro(cx, args) {
                return;
            }

            self.format_args.insert(expr.span.with_parent(None), (**args).clone());
        }
    }

    fn check_crate_post(&mut self, _: &EarlyContext<'_>, _: &Crate) {
        self.storage.set(mem::take(&mut self.format_args));
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
    !once(args.span)
        .chain(argument_span)
        .tuple_windows()
        .map(|(start, end)| start.between(end))
        .all(|sp| {
            sp.check_source_text(cx, |src| {
                // text should be either `, name` or `, name =`
                let mut iter = tokenize(src, FrontmatterAllowed::No).filter(|t| {
                    !matches!(
                        t.kind,
                        TokenKind::LineComment { .. } | TokenKind::BlockComment { .. } | TokenKind::Whitespace
                    )
                });
                iter.next().is_some_and(|t| matches!(t.kind, TokenKind::Comma))
                    && iter.all(|t| matches!(t.kind, TokenKind::Ident | TokenKind::Eq))
            })
        })
}
