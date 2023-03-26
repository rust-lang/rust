use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{root_macro_call_first_node, FormatArgsExpn};
use clippy_utils::source::snippet_with_context;
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
    /// let foo = "foo";
    /// format!("{}", foo);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let foo = "foo";
    /// foo.to_owned();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_FORMAT,
    complexity,
    "useless use of `format!`"
}

declare_lint_pass!(UselessFormat => [USELESS_FORMAT]);

impl<'tcx> LateLintPass<'tcx> for UselessFormat {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (format_args, call_site) = if_chain! {
            if let Some(macro_call) = root_macro_call_first_node(cx, expr);
            if cx.tcx.is_diagnostic_item(sym::format_macro, macro_call.def_id);
            if let Some(format_args) = FormatArgsExpn::find_nested(cx, expr, macro_call.expn);
            then {
                (format_args, macro_call.span)
            } else {
                return
            }
        };

        let mut applicability = Applicability::MachineApplicable;
        if format_args.args.is_empty() {
            match *format_args.format_string.parts {
                [] => span_useless_format_empty(cx, call_site, "String::new()".to_owned(), applicability),
                [_] => {
                    // Simulate macro expansion, converting {{ and }} to { and }.
                    let s_expand = format_args.format_string.snippet.replace("{{", "{").replace("}}", "}");
                    let sugg = format!("{s_expand}.to_string()");
                    span_useless_format(cx, call_site, sugg, applicability);
                },
                [..] => {},
            }
        } else if let [arg] = &*format_args.args {
            let value = arg.param.value;
            if_chain! {
                if format_args.format_string.parts == [kw::Empty];
                if arg.format.is_default();
                if match cx.typeck_results().expr_ty(value).peel_refs().kind() {
                    ty::Adt(adt, _) => Some(adt.did()) == cx.tcx.lang_items().string(),
                    ty::Str => true,
                    _ => false,
                };
                then {
                    let is_new_string = match value.kind {
                        ExprKind::Binary(..) => true,
                        ExprKind::MethodCall(path, ..) => path.ident.name == sym::to_string,
                        _ => false,
                    };
                    let sugg = if is_new_string {
                        snippet_with_context(cx, value.span, call_site.ctxt(), "..", &mut applicability).0.into_owned()
                    } else {
                        let sugg = Sugg::hir_with_context(cx, value, call_site.ctxt(), "<arg>", &mut applicability);
                        format!("{}.to_string()", sugg.maybe_par())
                    };
                    span_useless_format(cx, call_site, sugg, applicability);
                }
            }
        };
    }
}

fn span_useless_format_empty(cx: &LateContext<'_>, span: Span, sugg: String, applicability: Applicability) {
    span_lint_and_sugg(
        cx,
        USELESS_FORMAT,
        span,
        "useless use of `format!`",
        "consider using `String::new()`",
        sugg,
        applicability,
    );
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
