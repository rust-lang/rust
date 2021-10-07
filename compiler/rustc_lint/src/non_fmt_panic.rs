use crate::{LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_errors::{pluralize, Applicability};
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_parse_format::{ParseMode, Parser, Piece};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_span::edition::Edition;
use rustc_span::{hygiene, sym, symbol::kw, symbol::SymbolStr, InnerSpan, Span, Symbol};
use rustc_trait_selection::infer::InferCtxtExt;

declare_lint! {
    /// The `non_fmt_panics` lint detects `panic!(..)` invocations where the first
    /// argument is not a formatting string.
    ///
    /// ### Example
    ///
    /// ```rust,no_run,edition2018
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
    NON_FMT_PANICS,
    Warn,
    "detect single-argument panic!() invocations in which the argument is not a format string",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2021),
        explain_reason: false,
    };
    report_in_external_macro
}

declare_lint_pass!(NonPanicFmt => [NON_FMT_PANICS]);

impl<'tcx> LateLintPass<'tcx> for NonPanicFmt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(f, [arg]) = &expr.kind {
            if let &ty::FnDef(def_id, _) = cx.typeck_results().expr_ty(f).kind() {
                if Some(def_id) == cx.tcx.lang_items().begin_panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_fn()
                    || Some(def_id) == cx.tcx.lang_items().panic_str()
                {
                    if let Some(id) = f.span.ctxt().outer_expn_data().macro_def_id {
                        if matches!(
                            cx.tcx.get_diagnostic_name(id),
                            Some(sym::core_panic_2015_macro | sym::std_panic_2015_macro)
                        ) {
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

    let (span, panic, symbol_str) = panic_call(cx, f);

    if in_external_macro(cx.sess(), span) {
        // Nothing that can be done about it in the current crate.
        return;
    }

    // Find the span of the argument to `panic!()`, before expansion in the
    // case of `panic!(some_macro!())`.
    // We don't use source_callsite(), because this `panic!(..)` might itself
    // be expanded from another macro, in which case we want to stop at that
    // expansion.
    let mut arg_span = arg.span;
    let mut arg_macro = None;
    while !span.contains(arg_span) {
        let expn = arg_span.ctxt().outer_expn_data();
        if expn.is_root() {
            break;
        }
        arg_macro = expn.macro_def_id;
        arg_span = expn.call_site;
    }

    cx.struct_span_lint(NON_FMT_PANICS, arg_span, |lint| {
        let mut l = lint.build("panic message is not a string literal");
        l.note(&format!("this usage of {}!() is deprecated; it will be a hard error in Rust 2021", symbol_str));
        l.note("for more information, see <https://doc.rust-lang.org/nightly/edition-guide/rust-2021/panic-macro-consistency.html>");
        if !is_arg_inside_call(arg_span, span) {
            // No clue where this argument is coming from.
            l.emit();
            return;
        }
        if arg_macro.map_or(false, |id| cx.tcx.is_diagnostic_item(sym::format_macro, id)) {
            // A case of `panic!(format!(..))`.
            l.note(format!("the {}!() macro supports formatting, so there's no need for the format!() macro here", symbol_str).as_str());
            if let Some((open, close, _)) = find_delimiters(cx, arg_span) {
                l.multipart_suggestion(
                    "remove the `format!(..)` macro call",
                    vec![
                        (arg_span.until(open.shrink_to_hi()), "".into()),
                        (close.until(arg_span.shrink_to_hi()), "".into()),
                    ],
                    Applicability::MachineApplicable,
                );
            }
        } else {
            let ty = cx.typeck_results().expr_ty(arg);
            // If this is a &str or String, we can confidently give the `"{}", ` suggestion.
            let is_str = matches!(
                ty.kind(),
                ty::Ref(_, r, _) if *r.kind() == ty::Str,
            ) || matches!(
                ty.ty_adt_def(),
                Some(ty_def) if cx.tcx.is_diagnostic_item(sym::String, ty_def.did),
            );

            let (suggest_display, suggest_debug) = cx.tcx.infer_ctxt().enter(|infcx| {
                let display = is_str || cx.tcx.get_diagnostic_item(sym::Display).map(|t| {
                    infcx.type_implements_trait(t, ty, InternalSubsts::empty(), cx.param_env).may_apply()
                }) == Some(true);
                let debug = !display && cx.tcx.get_diagnostic_item(sym::Debug).map(|t| {
                    infcx.type_implements_trait(t, ty, InternalSubsts::empty(), cx.param_env).may_apply()
                }) == Some(true);
                (display, debug)
            });

            let suggest_panic_any = !is_str && panic == sym::std_panic_macro;

            let fmt_applicability = if suggest_panic_any {
                // If we can use panic_any, use that as the MachineApplicable suggestion.
                Applicability::MaybeIncorrect
            } else {
                // If we don't suggest panic_any, using a format string is our best bet.
                Applicability::MachineApplicable
            };

            if suggest_display {
                l.span_suggestion_verbose(
                    arg_span.shrink_to_lo(),
                    "add a \"{}\" format string to Display the message",
                    "\"{}\", ".into(),
                    fmt_applicability,
                );
            } else if suggest_debug {
                l.span_suggestion_verbose(
                    arg_span.shrink_to_lo(),
                    &format!(
                        "add a \"{{:?}}\" format string to use the Debug implementation of `{}`",
                        ty,
                    ),
                    "\"{:?}\", ".into(),
                    fmt_applicability,
                );
            }

            if suggest_panic_any {
                if let Some((open, close, del)) = find_delimiters(cx, span) {
                    l.multipart_suggestion(
                        &format!(
                            "{}use std::panic::panic_any instead",
                            if suggest_display || suggest_debug {
                                "or "
                            } else {
                                ""
                            },
                        ),
                        if del == '(' {
                            vec![(span.until(open), "std::panic::panic_any".into())]
                        } else {
                            vec![
                                (span.until(open.shrink_to_hi()), "std::panic::panic_any(".into()),
                                (close, ")".into()),
                            ]
                        },
                        Applicability::MachineApplicable,
                    );
                }
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

    let (span, _, _) = panic_call(cx, f);

    if in_external_macro(cx.sess(), span) && in_external_macro(cx.sess(), arg.span) {
        // Nothing that can be done about it in the current crate.
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

    if n_arguments > 0 && fmt_parser.errors.is_empty() {
        let arg_spans: Vec<_> = match &fmt_parser.arg_places[..] {
            [] => vec![fmt_span],
            v => v.iter().map(|span| fmt_span.from_inner(*span)).collect(),
        };
        cx.struct_span_lint(NON_FMT_PANICS, arg_spans, |lint| {
            let mut l = lint.build(match n_arguments {
                1 => "panic message contains an unused formatting placeholder",
                _ => "panic message contains unused formatting placeholders",
            });
            l.note("this message is not used as a format string when given without arguments, but will be in Rust 2021");
            if is_arg_inside_call(arg.span, span) {
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
        cx.struct_span_lint(NON_FMT_PANICS, brace_spans.unwrap_or_else(|| vec![span]), |lint| {
            let mut l = lint.build(msg);
            l.note("this message is not used as a format string, but will be in Rust 2021");
            if is_arg_inside_call(arg.span, span) {
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

/// Given the span of `some_macro!(args);`, gives the span of `(` and `)`,
/// and the type of (opening) delimiter used.
fn find_delimiters<'tcx>(cx: &LateContext<'tcx>, span: Span) -> Option<(Span, Span, char)> {
    let snippet = cx.sess().parse_sess.source_map().span_to_snippet(span).ok()?;
    let (open, open_ch) = snippet.char_indices().find(|&(_, c)| "([{".contains(c))?;
    let close = snippet.rfind(|c| ")]}".contains(c))?;
    Some((
        span.from_inner(InnerSpan { start: open, end: open + 1 }),
        span.from_inner(InnerSpan { start: close, end: close + 1 }),
        open_ch,
    ))
}

fn panic_call<'tcx>(cx: &LateContext<'tcx>, f: &'tcx hir::Expr<'tcx>) -> (Span, Symbol, SymbolStr) {
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

    let macro_symbol =
        if let hygiene::ExpnKind::Macro(_, symbol) = expn.kind { symbol } else { sym::panic };
    (expn.call_site, panic_macro, macro_symbol.as_str())
}

fn is_arg_inside_call(arg: Span, call: Span) -> bool {
    // We only add suggestions if the argument we're looking at appears inside the
    // panic call in the source file, to avoid invalid suggestions when macros are involved.
    // We specifically check for the spans to not be identical, as that happens sometimes when
    // proc_macros lie about spans and apply the same span to all the tokens they produce.
    call.contains(arg) && !call.source_equal(&arg)
}
