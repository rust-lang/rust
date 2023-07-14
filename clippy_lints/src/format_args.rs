use arrayvec::ArrayVec;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::is_diag_trait_item;
use clippy_utils::macros::{
    find_format_arg_expr, find_format_args, format_arg_removal_span, format_placeholder_format_span, is_assert_macro,
    is_format_macro, is_panic, root_macro_call, root_macro_call_first_node, FormatParamUsage,
};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_type_lang_item};
use if_chain::if_chain;
use itertools::Itertools;
use rustc_ast::{
    FormatArgPosition, FormatArgPositionKind, FormatArgsPiece, FormatArgumentKind, FormatCount, FormatOptions,
    FormatPlaceholder, FormatTrait,
};
use rustc_errors::Applicability;
use rustc_errors::SuggestionStyle::{CompletelyHidden, ShowCode};
use rustc_hir::{Expr, ExprKind, LangItem};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::adjustment::{Adjust, Adjustment};
use rustc_middle::ty::Ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::DefId;
use rustc_span::edition::Edition::Edition2021;
use rustc_span::{sym, Span, Symbol};

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
    /// If allow-mixed-uninlined-format-args is set to false in clippy.toml,
    /// the following code will also trigger the lint:
    /// ```rust
    /// # let var = 42;
    /// format!("{} {}", var, 1+2);
    /// ```
    /// Use instead:
    /// ```rust
    /// # let var = 42;
    /// format!("{var} {}", 1+2);
    /// ```
    ///
    /// ### Known Problems
    ///
    /// If a format string contains a numbered argument that cannot be inlined
    /// nothing will be suggested, e.g. `println!("{0}={1}", var, 1+2)`.
    #[clippy::version = "1.66.0"]
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
    ignore_mixed: bool,
}

impl FormatArgs {
    #[must_use]
    pub fn new(msrv: Msrv, allow_mixed_uninlined_format_args: bool) -> Self {
        Self {
            msrv,
            ignore_mixed: allow_mixed_uninlined_format_args,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for FormatArgs {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        if !is_format_macro(cx, macro_call.def_id) {
            return;
        }
        let name = cx.tcx.item_name(macro_call.def_id);

        find_format_args(cx, expr, macro_call.expn, |format_args| {
            for piece in &format_args.template {
                if let FormatArgsPiece::Placeholder(placeholder) = piece
                    && let Ok(index) = placeholder.argument.index
                    && let Some(arg) = format_args.arguments.all_args().get(index)
                {
                    let arg_expr = find_format_arg_expr(expr, arg);

                    check_unused_format_specifier(cx, placeholder, arg_expr);

                    if placeholder.format_trait != FormatTrait::Display
                        || placeholder.format_options != FormatOptions::default()
                        || is_aliased(format_args, index)
                    {
                        continue;
                    }

                    if let Ok(arg_hir_expr) = arg_expr {
                        check_format_in_format_args(cx, macro_call.span, name, arg_hir_expr);
                        check_to_string_in_format_args(cx, name, arg_hir_expr);
                    }
                }
            }

            if self.msrv.meets(msrvs::FORMAT_ARGS_CAPTURE) {
                check_uninlined_args(cx, format_args, macro_call.span, macro_call.def_id, self.ignore_mixed);
            }
        });
    }

    extract_msrv_attr!(LateContext);
}

fn check_unused_format_specifier(
    cx: &LateContext<'_>,
    placeholder: &FormatPlaceholder,
    arg_expr: Result<&Expr<'_>, &rustc_ast::Expr>,
) {
    let ty_or_ast_expr = arg_expr.map(|expr| cx.typeck_results().expr_ty(expr).peel_refs());

    let is_format_args = match ty_or_ast_expr {
        Ok(ty) => is_type_lang_item(cx, ty, LangItem::FormatArguments),
        Err(expr) => matches!(expr.peel_parens_and_refs().kind, rustc_ast::ExprKind::FormatArgs(_)),
    };

    let options = &placeholder.format_options;

    let arg_span = match arg_expr {
        Ok(expr) => expr.span,
        Err(expr) => expr.span,
    };

    if let Some(placeholder_span) = placeholder.span
        && is_format_args
        && *options != FormatOptions::default()
    {
        span_lint_and_then(
            cx,
            UNUSED_FORMAT_SPECS,
            placeholder_span,
            "format specifiers have no effect on `format_args!()`",
            |diag| {
                let mut suggest_format = |spec| {
                    let message = format!("for the {spec} to apply consider using `format!()`");

                    if let Some(mac_call) = root_macro_call(arg_span)
                        && cx.tcx.is_diagnostic_item(sym::format_args_macro, mac_call.def_id)
                    {
                        diag.span_suggestion(
                            cx.sess().source_map().span_until_char(mac_call.span, '!'),
                            message,
                            "format",
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        diag.help(message);
                    }
                };

                if options.width.is_some() {
                    suggest_format("width");
                }

                if options.precision.is_some() {
                    suggest_format("precision");
                }

                if let Some(format_span) = format_placeholder_format_span(placeholder) {
                    diag.span_suggestion_verbose(
                        format_span,
                        "if the current behavior is intentional, remove the format specifiers",
                        "",
                        Applicability::MaybeIncorrect,
                    );
                }
            },
        );
    }
}

fn check_uninlined_args(
    cx: &LateContext<'_>,
    args: &rustc_ast::FormatArgs,
    call_site: Span,
    def_id: DefId,
    ignore_mixed: bool,
) {
    if args.span.from_expansion() {
        return;
    }
    if call_site.edition() < Edition2021 && (is_panic(cx, def_id) || is_assert_macro(cx, def_id)) {
        // panic!, assert!, and debug_assert! before 2021 edition considers a single string argument as
        // non-format
        return;
    }

    let mut fixes = Vec::new();
    // If any of the arguments are referenced by an index number,
    // and that argument is not a simple variable and cannot be inlined,
    // we cannot remove any other arguments in the format string,
    // because the index numbers might be wrong after inlining.
    // Example of an un-inlinable format:  print!("{}{1}", foo, 2)
    for (pos, usage) in format_arg_positions(args) {
        if !check_one_arg(args, pos, usage, &mut fixes, ignore_mixed) {
            return;
        }
    }

    if fixes.is_empty() {
        return;
    }

    // multiline span display suggestion is sometimes broken: https://github.com/rust-lang/rust/pull/102729#discussion_r988704308
    // in those cases, make the code suggestion hidden
    let multiline_fix = fixes.iter().any(|(span, _)| cx.sess().source_map().is_multiline(*span));

    // Suggest removing each argument only once, for example in `format!("{0} {0}", arg)`.
    fixes.sort_unstable_by_key(|(span, _)| *span);
    fixes.dedup_by_key(|(span, _)| *span);

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

fn check_one_arg(
    args: &rustc_ast::FormatArgs,
    pos: &FormatArgPosition,
    usage: FormatParamUsage,
    fixes: &mut Vec<(Span, String)>,
    ignore_mixed: bool,
) -> bool {
    let index = pos.index.unwrap();
    let arg = &args.arguments.all_args()[index];

    if !matches!(arg.kind, FormatArgumentKind::Captured(_))
        && let rustc_ast::ExprKind::Path(None, path) = &arg.expr.kind
        && let [segment] = path.segments.as_slice()
        && segment.args.is_none()
        && let Some(arg_span) = format_arg_removal_span(args, index)
        && let Some(pos_span) = pos.span
    {
        let replacement = match usage {
            FormatParamUsage::Argument => segment.ident.name.to_string(),
            FormatParamUsage::Width => format!("{}$", segment.ident.name),
            FormatParamUsage::Precision => format!(".{}$", segment.ident.name),
        };
        fixes.push((pos_span, replacement));
        fixes.push((arg_span, String::new()));
        true  // successful inlining, continue checking
    } else {
        // Do not continue inlining (return false) in case
        // * if we can't inline a numbered argument, e.g. `print!("{0} ...", foo.bar, ...)`
        // * if allow_mixed_uninlined_format_args is false and this arg hasn't been inlined already
        pos.kind != FormatArgPositionKind::Number
            && (!ignore_mixed || matches!(arg.kind, FormatArgumentKind::Captured(_)))
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
            diag.help(format!(
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

fn format_arg_positions(
    format_args: &rustc_ast::FormatArgs,
) -> impl Iterator<Item = (&FormatArgPosition, FormatParamUsage)> {
    format_args.template.iter().flat_map(|piece| match piece {
        FormatArgsPiece::Placeholder(placeholder) => {
            let mut positions = ArrayVec::<_, 3>::new();

            positions.push((&placeholder.argument, FormatParamUsage::Argument));
            if let Some(FormatCount::Argument(position)) = &placeholder.format_options.width {
                positions.push((position, FormatParamUsage::Width));
            }
            if let Some(FormatCount::Argument(position)) = &placeholder.format_options.precision {
                positions.push((position, FormatParamUsage::Precision));
            }

            positions
        },
        FormatArgsPiece::Literal(_) => ArrayVec::new(),
    })
}

/// Returns true if the format argument at `index` is referred to by multiple format params
fn is_aliased(format_args: &rustc_ast::FormatArgs, index: usize) -> bool {
    format_arg_positions(format_args)
        .filter(|(position, _)| position.index == Ok(index))
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
