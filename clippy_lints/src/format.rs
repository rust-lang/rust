use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::FormatExpn;
use clippy_utils::source::{snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
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
                    ty::Adt(adt, _) => cx.tcx.is_diagnostic_item(sym::String, adt.did),
                    ty::Str => true,
                    _ => false,
                };
                if let Some(args) = format_args.args();
                if args.iter().all(|arg| arg.is_display() && !arg.has_string_formatting());
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

fn span_useless_format(cx: &LateContext<'_>, span: Span, sugg: String, applicability: Applicability) {
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
