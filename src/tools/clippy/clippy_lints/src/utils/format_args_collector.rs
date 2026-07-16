use clippy_utils::macros::FormatArgsStorage;
use clippy_utils::source::{SpanExt, walk_span_to_context};
use rustc_ast::{Crate, Expr, ExprKind, FormatArgs};
use rustc_data_structures::fx::FxHashMap;
use rustc_lexer::{FrontmatterAllowed, TokenKind, tokenize};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::SourceMap;
use rustc_span::{Span, SpanData};
use std::mem;

/// Populates [`FormatArgsStorage`] with AST [`FormatArgs`] nodes
pub struct FormatArgsCollector {
    format_args: FxHashMap<Span, FormatArgs>,
    parent_spans: Vec<SpanData>,
    storage: FormatArgsStorage,
}

impl FormatArgsCollector {
    pub fn new(storage: FormatArgsStorage) -> Self {
        Self {
            format_args: FxHashMap::default(),
            parent_spans: Vec::new(),
            storage,
        }
    }
}

impl_lint_pass!(FormatArgsCollector => []);

impl EarlyLintPass for FormatArgsCollector {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let ExprKind::FormatArgs(args) = &expr.kind {
            if self.has_span_from_external_macro(cx.sess().source_map(), expr.span, args) {
                return;
            }

            self.format_args.insert(expr.span.with_parent(None), (**args).clone());
        }
    }

    fn check_crate_post(&mut self, _: &EarlyContext<'_>, _: &Crate) {
        self.storage.set(mem::take(&mut self.format_args));
    }
}

impl FormatArgsCollector {
    /// Detects if the format string or an argument has its span set by a proc macro to something
    /// inside a macro callsite, e.g.
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
    fn has_span_from_external_macro(&mut self, sm: &SourceMap, fmt_sp: Span, args: &FormatArgs) -> bool {
        let mut fmt_sp = fmt_sp.data();

        // Find the first macro call that contains the format string.
        let arg_sp = if let Some(arg_sp) = walk_span_to_context(args.span, fmt_sp.ctxt) {
            arg_sp.data()
        } else {
            // Try to find a common parent for the format call and the format string.
            self.parent_spans.clear();
            // `fmt_sp.ctxt` isn't a parent of the format string so don't add it to the
            // search. The first iteration will always run since it can't be the root.
            while !fmt_sp.ctxt.is_root() {
                fmt_sp = fmt_sp.ctxt.outer_expn_data().call_site.data();
                self.parent_spans.push(fmt_sp);
            }
            let mut arg_sp = args.span.data();
            // Note: A parent span will always eventually be found since the root context
            // is an ancestor of all contexts.
            loop {
                match self.parent_spans.iter().find(|s| s.ctxt == arg_sp.ctxt) {
                    Some(call_sp) if call_sp.lo <= arg_sp.lo && arg_sp.hi <= call_sp.hi => {
                        fmt_sp = *call_sp;
                        break arg_sp;
                    },
                    // If the string isn't within the call span we some macro stuff we can't
                    // easily interpret.
                    Some(_) => return true,
                    None => arg_sp = arg_sp.ctxt.outer_expn_data().call_site.data(),
                }
            }
        };
        if fmt_sp.ctxt.in_external_macro(sm) {
            return true;
        }
        let Some(src) = arg_sp.get_source_range(sm) else {
            return true;
        };
        let Some(src_text) = src.sf.src.as_ref().map(|x| &***x) else {
            return true;
        };

        // Check the spans between the format string and the arguments and between each argument.
        args.arguments
            .explicit_args()
            .iter()
            .try_fold(src.range.end, |start, arg| {
                let expr_sp = walk_span_to_context(arg.expr.span, fmt_sp.ctxt)?.data();
                let expr_start = (expr_sp.lo.0 - src.sf.start_pos.0) as usize;
                let expr_end = (expr_sp.hi.0 - src.sf.start_pos.0) as usize;
                let mut tks = tokenize(src_text.get(start..expr_start)?, FrontmatterAllowed::No)
                    .map(|x| x.kind)
                    .filter(|x| {
                        !matches!(
                            x,
                            TokenKind::LineComment { doc_style: None }
                                | TokenKind::BlockComment {
                                    doc_style: None,
                                    terminated: true
                                }
                                | TokenKind::Whitespace
                        )
                    });

                // `,` or `, ident =`
                let matches = matches!(tks.next(), Some(TokenKind::Comma))
                    && match tks.next() {
                        Some(TokenKind::Ident) => matches!(tks.next(), Some(TokenKind::Eq)),
                        Some(_) => false,
                        None => true,
                    }
                    && tks.next().is_none();
                matches.then_some(expr_end)
            })
            .is_none()
    }
}
