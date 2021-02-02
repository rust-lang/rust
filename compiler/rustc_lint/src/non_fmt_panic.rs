use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_errors::{pluralize, Applicability};
use rustc_hir as hir;
use rustc_middle::ty;
use rustc_parse_format::{ParseMode, Parser, Piece};
use rustc_span::{sym, symbol::kw, InnerSpan, Span, Symbol};

declare_lint! {
    /// The `non_fmt_panic` lint detects `panic!(..)` invocations where the first
    /// argument is not a formatting string.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// panic!("{}");
    /// panic!(123);
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In Rust 2018 and earlier, `panic!(x)` directly uses `x` as the message.
    /// That means that `panic!("{}")` panics with the message `"{}"` instead
    /// of using it as a formatting string, and `panic!(123)` will panic with
    /// an `i32` as message.
    ///
    /// Rust 2021 always interprets the first argument as format string.
    NON_FMT_PANIC,
    Warn,
    "detect single-argument panic!() invocations in which the argument is not a format string",
    report_in_external_macro
}

declare_lint_pass!(NonPanicFmt => [NON_FMT_PANIC]);

impl<'tcx> LateLintPass<'tcx> for NonPanicFmt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(f, [arg]) = &expr.kind {
            if let &ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(f).kind() {
                if Some(def_id) == cx.tcx.lang_items().begin_panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_str()
                {
                    if let Some(id) = f.span.ctxt().outer_expn_data().macro_def_id {
                        if cx.tcx.is_diagnostic_item(sym::std_panic_2015_macro, id)
                            || cx.tcx.is_diagnostic_item(sym::core_panic_2015_macro, id)
                        {
                            check_panic(cx, f, arg);
                        }
                    }
                }
            }
        }
    }
}

fn check_panic<'tcx>(cx: &LateContext<'tcx>, f: &'tcx hir::Expr<'tcx>, arg: &'tcx hir::Expr<'tcx>) {
    if let hir::ExprKind::Lit(lit) = &arg.kind {
        if let ast::LitKind::Str(sym, _) = lit.node {
            // The argument is a string literal.
            check_panic_str(cx, f, arg, &sym.as_str());
            return;
        }
    }

    // The argument is *not* a string literal.

    let (span, panic) = panic_call(cx, f);

    cx.struct_span_lint(NON_FMT_PANIC, arg.span, |lint| {
        let mut l = lint.build("panic message is not a string literal");
        l.note("this is no longer accepted in Rust 2021");
        if span.contains(arg.span) {
            l.span_suggestion_verbose(
                arg.span.shrink_to_lo(),
                "add a \"{}\" format string to Display the message",
                "\"{}\", ".into(),
                Applicability::MaybeIncorrect,
            );
            if panic == sym::std_panic_macro {
                l.span_suggestion_verbose(
                    span.until(arg.span),
                    "or use std::panic::panic_any instead",
                    "std::panic::panic_any(".into(),
                    Applicability::MachineApplicable,
                );
            }
        }
        l.emit();
    });
}

fn check_panic_str<'tcx>(
    cx: &LateContext<'tcx>,
    f: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'tcx>,
    fmt: &str,
) {
    if !fmt.contains(&['{', '}'][..]) {
        // No brace, no problem.
        return;
    }

    let fmt_span = arg.span.source_callsite();

    let (snippet, style) = match cx.sess().parse_sess.source_map().span_to_snippet(fmt_span) {
        Ok(snippet) => {
            // Count the number of `#`s between the `r` and `"`.
            let style = snippet.strip_prefix('r').and_then(|s| s.find('"'));
            (Some(snippet), style)
        }
        Err(_) => (None, None),
    };

    let mut fmt_parser =
        Parser::new(fmt.as_ref(), style, snippet.clone(), false, ParseMode::Format);
    let n_arguments = (&mut fmt_parser).filter(|a| matches!(a, Piece::NextArgument(_))).count();

    let (span, _) = panic_call(cx, f);

    if n_arguments > 0 && fmt_parser.errors.is_empty() {
        let arg_spans: Vec<_> = match &fmt_parser.arg_places[..] {
            [] => vec![fmt_span],
            v => v.iter().map(|span| fmt_span.from_inner(*span)).collect(),
        };
        cx.struct_span_lint(NON_FMT_PANIC, arg_spans, |lint| {
            let mut l = lint.build(match n_arguments {
                1 => "panic message contains an unused formatting placeholder",
                _ => "panic message contains unused formatting placeholders",
            });
            l.note("this message is not used as a format string when given without arguments, but will be in Rust 2021");
            if span.contains(arg.span) {
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
        let brace_spans: Option<Vec<_>> =
            snippet.filter(|s| s.starts_with('"') || s.starts_with("r#")).map(|s| {
                s.char_indices()
                    .filter(|&(_, c)| c == '{' || c == '}')
                    .map(|(i, _)| fmt_span.from_inner(InnerSpan { start: i, end: i + 1 }))
                    .collect()
            });
        let msg = match &brace_spans {
            Some(v) if v.len() == 1 => "panic message contains a brace",
            _ => "panic message contains braces",
        };
        cx.struct_span_lint(NON_FMT_PANIC, brace_spans.unwrap_or(vec![span]), |lint| {
            let mut l = lint.build(msg);
            l.note("this message is not used as a format string, but will be in Rust 2021");
            if span.contains(arg.span) {
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

fn panic_call<'tcx>(cx: &LateContext<'tcx>, f: &'tcx hir::Expr<'tcx>) -> (Span, Symbol) {
    let mut expn = f.span.ctxt().outer_expn_data();

    let mut panic_macro = kw::Empty;

    // Unwrap more levels of macro expansion, as panic_2015!()
    // was likely expanded from panic!() and possibly from
    // [debug_]assert!().
    for &i in
        &[sym::std_panic_macro, sym::core_panic_macro, sym::assert_macro, sym::debug_assert_macro]
    {
        let parent = expn.call_site.ctxt().outer_expn_data();
        if parent.macro_def_id.map_or(false, |id| cx.tcx.is_diagnostic_item(i, id)) {
            expn = parent;
            panic_macro = i;
        }
    }

    (expn.call_site, panic_macro)
}
