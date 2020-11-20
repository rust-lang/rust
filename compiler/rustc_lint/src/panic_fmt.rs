use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_errors::{pluralize, Applicability};
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_parse_format::{ParseMode, Parser, Piece};
use rustc_span::{sym, InnerSpan};

declare_lint! {
    /// The `panic_fmt` lint detects `panic!("..")` with `{` or `}` in the string literal.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// panic!("{}");
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `panic!("{}")` panics with the message `"{}"`, as a `panic!()` invocation
    /// with a single argument does not use `format_args!()`.
    /// A future edition of Rust will interpret this string as format string,
    /// which would break this.
    PANIC_FMT,
    Warn,
    "detect braces in single-argument panic!() invocations",
    report_in_external_macro
}

declare_lint_pass!(PanicFmt => [PANIC_FMT]);

impl<'tcx> LateLintPass<'tcx> for PanicFmt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(f, [arg]) = &expr.kind {
            if let &ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(f).kind() {
                if Some(def_id) == cx.tcx.lang_items().begin_panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_fn()
                {
                    check_panic(cx, f, arg);
                }
            }
        }
    }
}

fn check_panic<'tcx>(cx: &LateContext<'tcx>, f: &'tcx hir::Expr<'tcx>, arg: &'tcx hir::Expr<'tcx>) {
    if let hir::ExprKind::Lit(lit) = &arg.kind {
        if let ast::LitKind::Str(sym, _) = lit.node {
            let mut expn = f.span.ctxt().outer_expn_data();
            if let Some(id) = expn.macro_def_id {
                if cx.tcx.is_diagnostic_item(sym::std_panic_macro, id)
                    || cx.tcx.is_diagnostic_item(sym::core_panic_macro, id)
                {
                    let fmt = sym.as_str();
                    if !fmt.contains(&['{', '}'][..]) {
                        return;
                    }

                    let fmt_span = arg.span.source_callsite();

                    let (snippet, style) =
                        match cx.sess().parse_sess.source_map().span_to_snippet(fmt_span) {
                            Ok(snippet) => {
                                // Count the number of `#`s between the `r` and `"`.
                                let style = snippet.strip_prefix('r').and_then(|s| s.find('"'));
                                (Some(snippet), style)
                            }
                            Err(_) => (None, None),
                        };

                    let mut fmt_parser =
                        Parser::new(fmt.as_ref(), style, snippet.clone(), false, ParseMode::Format);
                    let n_arguments =
                        (&mut fmt_parser).filter(|a| matches!(a, Piece::NextArgument(_))).count();

                    // Unwrap another level of macro expansion if this panic!()
                    // was expanded from assert!() or debug_assert!().
                    for &assert in &[sym::assert_macro, sym::debug_assert_macro] {
                        let parent = expn.call_site.ctxt().outer_expn_data();
                        if parent
                            .macro_def_id
                            .map_or(false, |id| cx.tcx.is_diagnostic_item(assert, id))
                        {
                            expn = parent;
                        }
                    }

                    if n_arguments > 0 && fmt_parser.errors.is_empty() {
                        let arg_spans: Vec<_> = match &fmt_parser.arg_places[..] {
                            [] => vec![fmt_span],
                            v => v.iter().map(|span| fmt_span.from_inner(*span)).collect(),
                        };
                        cx.struct_span_lint(PANIC_FMT, arg_spans, |lint| {
                            let mut l = lint.build(match n_arguments {
                                1 => "panic message contains an unused formatting placeholder",
                                _ => "panic message contains unused formatting placeholders",
                            });
                            l.note("this message is not used as a format string when given without arguments, but will be in a future Rust edition");
                            if expn.call_site.contains(arg.span) {
                                l.span_suggestion(
                                    arg.span.shrink_to_hi(),
                                    &format!("add the missing argument{}", pluralize!(n_arguments)),
                                    ", ...".into(),
                                    Applicability::HasPlaceholders,
                                );
                                l.span_suggestion(
                                    arg.span.shrink_to_lo(),
                                    "or add a \"{}\" format string to use the message literally",
                                    "\"{}\", ".into(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            l.emit();
                        });
                    } else {
                        let brace_spans: Option<Vec<_>> = snippet
                            .filter(|s| s.starts_with('"') || s.starts_with("r#"))
                            .map(|s| {
                                s.char_indices()
                                    .filter(|&(_, c)| c == '{' || c == '}')
                                    .map(|(i, _)| {
                                        fmt_span.from_inner(InnerSpan { start: i, end: i + 1 })
                                    })
                                    .collect()
                            });
                        let msg = match &brace_spans {
                            Some(v) if v.len() == 1 => "panic message contains a brace",
                            _ => "panic message contains braces",
                        };
                        cx.struct_span_lint(PANIC_FMT, brace_spans.unwrap_or(vec![expn.call_site]), |lint| {
                            let mut l = lint.build(msg);
                            l.note("this message is not used as a format string, but will be in a future Rust edition");
                            if expn.call_site.contains(arg.span) {
                                l.span_suggestion(
                                    arg.span.shrink_to_lo(),
                                    "add a \"{}\" format string to use the message literally",
                                    "\"{}\", ".into(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            l.emit();
                        });
                    }
                }
            }
        }
    }
}
