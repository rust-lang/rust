use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::FormatExpn;
use clippy_utils::last_path_segment;
use clippy_utils::source::{snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of `format!("string literal with no
    /// argument")` and `format!("{}", foo)` where `foo` is a string.
    ///
    /// ### Why is this bad?
    /// There is no point of doing that. `format!("foo")` can
    /// be replaced by `"foo".to_owned()` if you really need a `String`. The even
    /// worse `&format!("foo")` is often encountered in the wild. `format!("{}",
    /// foo)` can be replaced by `foo.clone()` if `foo: String` or `foo.to_owned()`
    /// if `foo: &str`.
    ///
    /// ### Examples
    /// ```rust
    ///
    /// // Bad
    /// let foo = "foo";
    /// format!("{}", foo);
    ///
    /// // Good
    /// foo.to_owned();
    /// ```
    pub USELESS_FORMAT,
    complexity,
    "useless use of `format!`"
}

declare_lint_pass!(UselessFormat => [USELESS_FORMAT]);

impl<'tcx> LateLintPass<'tcx> for UselessFormat {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let FormatExpn { call_site, format_args } = match FormatExpn::parse(expr) {
            Some(e) if !e.call_site.from_expansion() => e,
            _ => return,
        };

        let mut applicability = Applicability::MachineApplicable;
        if format_args.value_args.is_empty() {
            if_chain! {
                if let [e] = &*format_args.format_string_parts;
                if let ExprKind::Lit(lit) = &e.kind;
                if let Some(s_src) = snippet_opt(cx, lit.span);
                then {
                    // Simulate macro expansion, converting {{ and }} to { and }.
                    let s_expand = s_src.replace("{{", "{").replace("}}", "}");
                    let sugg = format!("{}.to_string()", s_expand);
                    span_useless_format(cx, call_site, sugg, applicability);
                }
            }
        } else if let [value] = *format_args.value_args {
            if_chain! {
                if format_args.format_string_symbols == [kw::Empty];
                if match cx.typeck_results().expr_ty(value).peel_refs().kind() {
                    ty::Adt(adt, _) => cx.tcx.is_diagnostic_item(sym::string_type, adt.did),
                    ty::Str => true,
                    _ => false,
                };
                if format_args.args.iter().all(is_display_arg);
                if format_args.fmt_expr.map_or(true, check_unformatted);
                then {
                    let is_new_string = match value.kind {
                        ExprKind::Binary(..) => true,
                        ExprKind::MethodCall(path, ..) => path.ident.name.as_str() == "to_string",
                        _ => false,
                    };
                    let sugg = if is_new_string {
                        snippet_with_applicability(cx, value.span, "..", &mut applicability).into_owned()
                    } else {
                        let sugg = Sugg::hir_with_applicability(cx, value, "<arg>", &mut applicability);
                        format!("{}.to_string()", sugg.maybe_par())
                    };
                    span_useless_format(cx, call_site, sugg, applicability);
                }
            }
        };
    }
}

fn span_useless_format(cx: &LateContext<'_>, span: Span, mut sugg: String, mut applicability: Applicability) {
    // The callsite span contains the statement semicolon for some reason.
    if snippet_with_applicability(cx, span, "..", &mut applicability).ends_with(';') {
        sugg.push(';');
    }

    span_lint_and_sugg(
        cx,
        USELESS_FORMAT,
        span,
        "useless use of `format!`",
        "consider using `.to_string()`",
        sugg,
        applicability,
    );
}

fn is_display_arg(expr: &Expr<'_>) -> bool {
    if_chain! {
        if let ExprKind::Call(_, [_, fmt]) = expr.kind;
        if let ExprKind::Path(QPath::Resolved(_, path)) = fmt.kind;
        if let [.., t, _] = path.segments;
        if t.ident.name.as_str() == "Display";
        then { true } else { false }
    }
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
        if let ExprKind::AddrOf(BorrowKind::Ref, _, expr) = expr.kind;
        if let ExprKind::Array([expr]) = expr.kind;
        // struct `core::fmt::rt::v1::Argument`
        if let ExprKind::Struct(_, fields, _) = expr.kind;
        if let Some(format_field) = fields.iter().find(|f| f.ident.name == sym::format);
        // struct `core::fmt::rt::v1::FormatSpec`
        if let ExprKind::Struct(_, fields, _) = format_field.expr.kind;
        if let Some(precision_field) = fields.iter().find(|f| f.ident.name == sym::precision);
        if let ExprKind::Path(ref precision_path) = precision_field.expr.kind;
        if last_path_segment(precision_path).ident.name == sym::Implied;
        if let Some(width_field) = fields.iter().find(|f| f.ident.name == sym::width);
        if let ExprKind::Path(ref width_qpath) = width_field.expr.kind;
        if last_path_segment(width_qpath).ident.name == sym::Implied;
        then {
            return true;
        }
    }

    false
}
