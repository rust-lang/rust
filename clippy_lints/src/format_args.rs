use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::is_diag_trait_item;
use clippy_utils::macros::FormatParamKind::{Implicit, Named, Numbered, Starred};
use clippy_utils::macros::{
    is_format_macro, is_panic, root_macro_call, Count, FormatArg, FormatArgsExpn, FormatParam, FormatParamUsage,
};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;
use itertools::Itertools;
use rustc_errors::{
    Applicability,
    SuggestionStyle::{CompletelyHidden, ShowCode},
};
use rustc_hir::{Expr, ExprKind, HirId, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::Ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::DefId;
use rustc_span::edition::Edition::Edition2021;
use rustc_span::{sym, ExpnData, ExpnKind, Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Detects `format!` within the arguments of another macro that does
    /// formatting such as `format!` itself, `write!` or `println!`. Suggests
    /// inlining the `format!` call.
    ///
    /// ### Why is this bad?
    /// The recommended code is both shorter and avoids a temporary allocation.
    ///
    /// ### Example
    /// ```rust
    /// # use std::panic::Location;
    /// println!("error: {}", format!("something failed at {}", Location::caller()));
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::panic::Location;
    /// println!("error: something failed at {}", Location::caller());
    /// ```
    #[clippy::version = "1.58.0"]
    pub FORMAT_IN_FORMAT_ARGS,
    perf,
    "`format!` used in a macro that does formatting"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for [`ToString::to_string`](https://doc.rust-lang.org/std/string/trait.ToString.html#tymethod.to_string)
    /// applied to a type that implements [`Display`](https://doc.rust-lang.org/std/fmt/trait.Display.html)
    /// in a macro that does formatting.
    ///
    /// ### Why is this bad?
    /// Since the type implements `Display`, the use of `to_string` is
    /// unnecessary.
    ///
    /// ### Example
    /// ```rust
    /// # use std::panic::Location;
    /// println!("error: something failed at {}", Location::caller().to_string());
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::panic::Location;
    /// println!("error: something failed at {}", Location::caller());
    /// ```
    #[clippy::version = "1.58.0"]
    pub TO_STRING_IN_FORMAT_ARGS,
    perf,
    "`to_string` applied to a type that implements `Display` in format args"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detect when a variable is not inlined in a format string,
    /// and suggests to inline it.
    ///
    /// ### Why is this bad?
    /// Non-inlined code is slightly more difficult to read and understand,
    /// as it requires arguments to be matched against the format string.
    /// The inlined syntax, where allowed, is simpler.
    ///
    /// ### Example
    /// ```rust
    /// # let var = 42;
    /// # let width = 1;
    /// # let prec = 2;
    /// format!("{}", var);
    /// format!("{v:?}", v = var);
    /// format!("{0} {0}", var);
    /// format!("{0:1$}", var, width);
    /// format!("{:.*}", prec, var);
    /// ```
    /// Use instead:
    /// ```rust
    /// # let var = 42;
    /// # let width = 1;
    /// # let prec = 2;
    /// format!("{var}");
    /// format!("{var:?}");
    /// format!("{var} {var}");
    /// format!("{var:width$}");
    /// format!("{var:.prec$}");
    /// ```
    ///
    /// ### Known Problems
    ///
    /// There may be a false positive if the format string is expanded from certain proc macros:
    ///
    /// ```ignore
    /// println!(indoc!("{}"), var);
    /// ```
    ///
    /// If a format string contains a numbered argument that cannot be inlined
    /// nothing will be suggested, e.g. `println!("{0}={1}", var, 1+2)`.
    #[clippy::version = "1.65.0"]
    pub UNINLINED_FORMAT_ARGS,
    pedantic,
    "using non-inlined variables in `format!` calls"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects [formatting parameters] that have no effect on the output of
    /// `format!()`, `println!()` or similar macros.
    ///
    /// ### Why is this bad?
    /// Shorter format specifiers are easier to read, it may also indicate that
    /// an expected formatting operation such as adding padding isn't happening.
    ///
    /// ### Example
    /// ```rust
    /// println!("{:.}", 1.0);
    ///
    /// println!("not padded: {:5}", format_args!("..."));
    /// ```
    /// Use instead:
    /// ```rust
    /// println!("{}", 1.0);
    ///
    /// println!("not padded: {}", format_args!("..."));
    /// // OR
    /// println!("padded: {:5}", format!("..."));
    /// ```
    ///
    /// [formatting parameters]: https://doc.rust-lang.org/std/fmt/index.html#formatting-parameters
    #[clippy::version = "1.66.0"]
    pub UNUSED_FORMAT_SPECS,
    complexity,
    "use of a format specifier that has no effect"
}

impl_lint_pass!(FormatArgs => [
    FORMAT_IN_FORMAT_ARGS,
    TO_STRING_IN_FORMAT_ARGS,
    UNINLINED_FORMAT_ARGS,
    UNUSED_FORMAT_SPECS,
]);

pub struct FormatArgs {
    msrv: Msrv,
}

impl FormatArgs {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for FormatArgs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(format_args) = FormatArgsExpn::parse(cx, expr)
            && let expr_expn_data = expr.span.ctxt().outer_expn_data()
            && let outermost_expn_data = outermost_expn_data(expr_expn_data)
            && let Some(macro_def_id) = outermost_expn_data.macro_def_id
            && is_format_macro(cx, macro_def_id)
            && let ExpnKind::Macro(_, name) = outermost_expn_data.kind
        {
            for arg in &format_args.args {
                check_unused_format_specifier(cx, arg);
                if !arg.format.is_default() {
                    continue;
                }
                if is_aliased(&format_args, arg.param.value.hir_id) {
                    continue;
                }
                check_format_in_format_args(cx, outermost_expn_data.call_site, name, arg.param.value);
                check_to_string_in_format_args(cx, name, arg.param.value);
            }
            if self.msrv.meets(msrvs::FORMAT_ARGS_CAPTURE) {
                check_uninlined_args(cx, &format_args, outermost_expn_data.call_site, macro_def_id);
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn check_unused_format_specifier(cx: &LateContext<'_>, arg: &FormatArg<'_>) {
    let param_ty = cx.typeck_results().expr_ty(arg.param.value).peel_refs();

    if let Count::Implied(Some(mut span)) = arg.format.precision
        && !span.is_empty()
    {
        span_lint_and_then(
            cx,
            UNUSED_FORMAT_SPECS,
            span,
            "empty precision specifier has no effect",
            |diag| {
                if param_ty.is_floating_point() {
                    diag.note("a precision specifier is not required to format floats");
                }

                if arg.format.is_default() {
                    // If there's no other specifiers remove the `:` too
                    span = arg.format_span();
                }

                diag.span_suggestion_verbose(span, "remove the `.`", "", Applicability::MachineApplicable);
            },
        );
    }

    if is_type_diagnostic_item(cx, param_ty, sym::Arguments) && !arg.format.is_default_for_trait() {
        span_lint_and_then(
            cx,
            UNUSED_FORMAT_SPECS,
            arg.span,
            "format specifiers have no effect on `format_args!()`",
            |diag| {
                let mut suggest_format = |spec, span| {
                    let message = format!("for the {spec} to apply consider using `format!()`");

                    if let Some(mac_call) = root_macro_call(arg.param.value.span)
                        && cx.tcx.is_diagnostic_item(sym::format_args_macro, mac_call.def_id)
                        && arg.span.eq_ctxt(mac_call.span)
                    {
                        diag.span_suggestion(
                            cx.sess().source_map().span_until_char(mac_call.span, '!'),
                            message,
                            "format",
                            Applicability::MaybeIncorrect,
                        );
                    } else if let Some(span) = span {
                        diag.span_help(span, message);
                    }
                };

                if !arg.format.width.is_implied() {
                    suggest_format("width", arg.format.width.span());
                }

                if !arg.format.precision.is_implied() {
                    suggest_format("precision", arg.format.precision.span());
                }

                diag.span_suggestion_verbose(
                    arg.format_span(),
                    "if the current behavior is intentional, remove the format specifiers",
                    "",
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}

fn check_uninlined_args(cx: &LateContext<'_>, args: &FormatArgsExpn<'_>, call_site: Span, def_id: DefId) {
    if args.format_string.span.from_expansion() {
        return;
    }
    if call_site.edition() < Edition2021 && is_panic(cx, def_id) {
        // panic! before 2021 edition considers a single string argument as non-format
        return;
    }

    let mut fixes = Vec::new();
    // If any of the arguments are referenced by an index number,
    // and that argument is not a simple variable and cannot be inlined,
    // we cannot remove any other arguments in the format string,
    // because the index numbers might be wrong after inlining.
    // Example of an un-inlinable format:  print!("{}{1}", foo, 2)
    if !args.params().all(|p| check_one_arg(args, &p, &mut fixes)) || fixes.is_empty() {
        return;
    }

    // multiline span display suggestion is sometimes broken: https://github.com/rust-lang/rust/pull/102729#discussion_r988704308
    // in those cases, make the code suggestion hidden
    let multiline_fix = fixes.iter().any(|(span, _)| cx.sess().source_map().is_multiline(*span));

    span_lint_and_then(
        cx,
        UNINLINED_FORMAT_ARGS,
        call_site,
        "variables can be used directly in the `format!` string",
        |diag| {
            diag.multipart_suggestion_with_style(
                "change this to",
                fixes,
                Applicability::MachineApplicable,
                if multiline_fix { CompletelyHidden } else { ShowCode },
            );
        },
    );
}

fn check_one_arg(args: &FormatArgsExpn<'_>, param: &FormatParam<'_>, fixes: &mut Vec<(Span, String)>) -> bool {
    if matches!(param.kind, Implicit | Starred | Named(_) | Numbered)
        && let ExprKind::Path(QPath::Resolved(None, path)) = param.value.kind
        && let [segment] = path.segments
        && let Some(arg_span) = args.value_with_prev_comma_span(param.value.hir_id)
    {
        let replacement = match param.usage {
            FormatParamUsage::Argument => segment.ident.name.to_string(),
            FormatParamUsage::Width => format!("{}$", segment.ident.name),
            FormatParamUsage::Precision => format!(".{}$", segment.ident.name),
        };
        fixes.push((param.span, replacement));
        fixes.push((arg_span, String::new()));
        true  // successful inlining, continue checking
    } else {
        // if we can't inline a numbered argument, we can't continue
        param.kind != Numbered
    }
}

fn outermost_expn_data(expn_data: ExpnData) -> ExpnData {
    if expn_data.call_site.from_expansion() {
        outermost_expn_data(expn_data.call_site.ctxt().outer_expn_data())
    } else {
        expn_data
    }
}

fn check_format_in_format_args(cx: &LateContext<'_>, call_site: Span, name: Symbol, arg: &Expr<'_>) {
    let expn_data = arg.span.ctxt().outer_expn_data();
    if expn_data.call_site.from_expansion() {
        return;
    }
    let Some(mac_id) = expn_data.macro_def_id else { return };
    if !cx.tcx.is_diagnostic_item(sym::format_macro, mac_id) {
        return;
    }
    span_lint_and_then(
        cx,
        FORMAT_IN_FORMAT_ARGS,
        call_site,
        &format!("`format!` in `{name}!` args"),
        |diag| {
            diag.help(&format!(
                "combine the `format!(..)` arguments with the outer `{name}!(..)` call"
            ));
            diag.help("or consider changing `format!` to `format_args!`");
        },
    );
}

fn check_to_string_in_format_args(cx: &LateContext<'_>, name: Symbol, value: &Expr<'_>) {
    if_chain! {
        if !value.span.from_expansion();
        if let ExprKind::MethodCall(_, receiver, [], to_string_span) = value.kind;
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(value.hir_id);
        if is_diag_trait_item(cx, method_def_id, sym::ToString);
        let receiver_ty = cx.typeck_results().expr_ty(receiver);
        if let Some(display_trait_id) = cx.tcx.get_diagnostic_item(sym::Display);
        let (n_needed_derefs, target) =
            count_needed_derefs(receiver_ty, cx.typeck_results().expr_adjustments(receiver).iter());
        if implements_trait(cx, target, display_trait_id, &[]);
        if let Some(sized_trait_id) = cx.tcx.lang_items().sized_trait();
        if let Some(receiver_snippet) = snippet_opt(cx, receiver.span);
        then {
            let needs_ref = !implements_trait(cx, receiver_ty, sized_trait_id, &[]);
            if n_needed_derefs == 0 && !needs_ref {
                span_lint_and_sugg(
                    cx,
                    TO_STRING_IN_FORMAT_ARGS,
                    to_string_span.with_lo(receiver.span.hi()),
                    &format!(
                        "`to_string` applied to a type that implements `Display` in `{name}!` args"
                    ),
                    "remove this",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    TO_STRING_IN_FORMAT_ARGS,
                    value.span,
                    &format!(
                        "`to_string` applied to a type that implements `Display` in `{name}!` args"
                    ),
                    "use this",
                    format!(
                        "{}{:*>n_needed_derefs$}{receiver_snippet}",
                        if needs_ref { "&" } else { "" },
                        ""
                    ),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

/// Returns true if `hir_id` is referred to by multiple format params
fn is_aliased(args: &FormatArgsExpn<'_>, hir_id: HirId) -> bool {
    args.params()
        .filter(|param| param.value.hir_id == hir_id)
        .at_most_one()
        .is_err()
}

fn count_needed_derefs<'tcx, I>(mut ty: Ty<'tcx>, mut iter: I) -> (usize, Ty<'tcx>)
where
    I: Iterator<Item = &'tcx Adjustment<'tcx>>,
{
    let mut n_total = 0;
    let mut n_needed = 0;
    loop {
        if let Some(Adjustment {
            kind: Adjust::Deref(overloaded_deref),
            target,
        }) = iter.next()
        {
            n_total += 1;
            if overloaded_deref.is_some() {
                n_needed = n_total;
            }
            ty = *target;
        } else {
            return (n_needed, ty);
        }
    }
}
