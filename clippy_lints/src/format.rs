use crate::utils::paths;
use crate::utils::{
    is_expn_of, is_type_diagnostic_item, last_path_segment, match_def_path, match_function_call, snippet, snippet_opt,
    span_lint_and_then,
};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Arm, BorrowKind, Expr, ExprKind, MatchSource, PatKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for the use of `format!("string literal with no
    /// argument")` and `format!("{}", foo)` where `foo` is a string.
    ///
    /// **Why is this bad?** There is no point of doing that. `format!("foo")` can
    /// be replaced by `"foo".to_owned()` if you really need a `String`. The even
    /// worse `&format!("foo")` is often encountered in the wild. `format!("{}",
    /// foo)` can be replaced by `foo.clone()` if `foo: String` or `foo.to_owned()`
    /// if `foo: &str`.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
    /// ```rust
    ///
    /// // Bad
    /// # let foo = "foo";
    /// format!("{}", foo);
    ///
    /// // Good
    /// format!("foo");
    /// ```
    pub USELESS_FORMAT,
    complexity,
    "useless use of `format!`"
}

declare_lint_pass!(UselessFormat => [USELESS_FORMAT]);

impl<'tcx> LateLintPass<'tcx> for UselessFormat {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let span = match is_expn_of(expr.span, "format") {
            Some(s) if !s.from_expansion() => s,
            _ => return,
        };

        // Operate on the only argument of `alloc::fmt::format`.
        if let Some(sugg) = on_new_v1(cx, expr) {
            span_useless_format(cx, span, "consider using `.to_string()`", sugg);
        } else if let Some(sugg) = on_new_v1_fmt(cx, expr) {
            span_useless_format(cx, span, "consider using `.to_string()`", sugg);
        }
    }
}

fn span_useless_format<T: LintContext>(cx: &T, span: Span, help: &str, mut sugg: String) {
    let to_replace = span.source_callsite();

    // The callsite span contains the statement semicolon for some reason.
    let snippet = snippet(cx, to_replace, "..");
    if snippet.ends_with(';') {
        sugg.push(';');
    }

    span_lint_and_then(cx, USELESS_FORMAT, span, "useless use of `format!`", |diag| {
        diag.span_suggestion(
            to_replace,
            help,
            sugg,
            Applicability::MachineApplicable, // snippet
        );
    });
}

fn on_argumentv1_new<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arms: &'tcx [Arm<'_>]) -> Option<String> {
    if_chain! {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref format_args) = expr.kind;
        if let ExprKind::Array(ref elems) = arms[0].body.kind;
        if elems.len() == 1;
        if let Some(args) = match_function_call(cx, &elems[0], &paths::FMT_ARGUMENTV1_NEW);
        // matches `core::fmt::Display::fmt`
        if args.len() == 2;
        if let ExprKind::Path(ref qpath) = args[1].kind;
        if let Some(did) = cx.qpath_res(qpath, args[1].hir_id).opt_def_id();
        if match_def_path(cx, did, &paths::DISPLAY_FMT_METHOD);
        // check `(arg0,)` in match block
        if let PatKind::Tuple(ref pats, None) = arms[0].pat.kind;
        if pats.len() == 1;
        then {
            let ty = cx.typeck_results().pat_ty(&pats[0]).peel_refs();
            if *ty.kind() != rustc_middle::ty::Str && !is_type_diagnostic_item(cx, ty, sym!(string_type)) {
                return None;
            }
            if let ExprKind::Lit(ref lit) = format_args.kind {
                if let LitKind::Str(ref s, _) = lit.node {
                    return Some(format!("{:?}.to_string()", s.as_str()));
                }
            } else {
                let snip = snippet(cx, format_args.span, "<arg>");
                if let ExprKind::MethodCall(ref path, _, _, _) = format_args.kind {
                    if path.ident.name == sym!(to_string) {
                        return Some(format!("{}", snip));
                    }
                } else if let ExprKind::Binary(..) = format_args.kind {
                    return Some(format!("{}", snip));
                }
                return Some(format!("{}.to_string()", snip));
            }
        }
    }
    None
}

fn on_new_v1<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<String> {
    if_chain! {
        if let Some(args) = match_function_call(cx, expr, &paths::FMT_ARGUMENTS_NEW_V1);
        if args.len() == 2;
        // Argument 1 in `new_v1()`
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref arr) = args[0].kind;
        if let ExprKind::Array(ref pieces) = arr.kind;
        if pieces.len() == 1;
        if let ExprKind::Lit(ref lit) = pieces[0].kind;
        if let LitKind::Str(ref s, _) = lit.node;
        // Argument 2 in `new_v1()`
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref arg1) = args[1].kind;
        if let ExprKind::Match(ref matchee, ref arms, MatchSource::Normal) = arg1.kind;
        if arms.len() == 1;
        if let ExprKind::Tup(ref tup) = matchee.kind;
        then {
            // `format!("foo")` expansion contains `match () { () => [], }`
            if tup.is_empty() {
                if let Some(s_src) = snippet_opt(cx, lit.span) {
                    // Simulate macro expansion, converting {{ and }} to { and }.
                    let s_expand = s_src.replace("{{", "{").replace("}}", "}");
                    return Some(format!("{}.to_string()", s_expand))
                }
            } else if s.as_str().is_empty() {
                return on_argumentv1_new(cx, &tup[0], arms);
            }
        }
    }
    None
}

fn on_new_v1_fmt<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<String> {
    if_chain! {
        if let Some(args) = match_function_call(cx, expr, &paths::FMT_ARGUMENTS_NEW_V1_FORMATTED);
        if args.len() == 3;
        if check_unformatted(&args[2]);
        // Argument 1 in `new_v1_formatted()`
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref arr) = args[0].kind;
        if let ExprKind::Array(ref pieces) = arr.kind;
        if pieces.len() == 1;
        if let ExprKind::Lit(ref lit) = pieces[0].kind;
        if let LitKind::Str(..) = lit.node;
        // Argument 2 in `new_v1_formatted()`
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref arg1) = args[1].kind;
        if let ExprKind::Match(ref matchee, ref arms, MatchSource::Normal) = arg1.kind;
        if arms.len() == 1;
        if let ExprKind::Tup(ref tup) = matchee.kind;
        then {
            return on_argumentv1_new(cx, &tup[0], arms);
        }
    }
    None
}

/// Checks if the expression matches
/// ```rust,ignore
/// &[_ {
///    format: _ {
///         width: _::Implied,
///         precision: _::Implied,
///         ...
///    },
///    ...,
/// }]
/// ```
fn check_unformatted(expr: &Expr<'_>) -> bool {
    if_chain! {
        if let ExprKind::AddrOf(BorrowKind::Ref, _, ref expr) = expr.kind;
        if let ExprKind::Array(ref exprs) = expr.kind;
        if exprs.len() == 1;
        // struct `core::fmt::rt::v1::Argument`
        if let ExprKind::Struct(_, ref fields, _) = exprs[0].kind;
        if let Some(format_field) = fields.iter().find(|f| f.ident.name == sym!(format));
        // struct `core::fmt::rt::v1::FormatSpec`
        if let ExprKind::Struct(_, ref fields, _) = format_field.expr.kind;
        if let Some(precision_field) = fields.iter().find(|f| f.ident.name == sym!(precision));
        if let ExprKind::Path(ref precision_path) = precision_field.expr.kind;
        if last_path_segment(precision_path).ident.name == sym!(Implied);
        if let Some(width_field) = fields.iter().find(|f| f.ident.name == sym!(width));
        if let ExprKind::Path(ref width_qpath) = width_field.expr.kind;
        if last_path_segment(width_qpath).ident.name == sym!(Implied);
        then {
            return true;
        }
    }

    false
}
