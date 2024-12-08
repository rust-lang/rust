use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{FormatArgsStorage, find_format_arg_expr, root_macro_call_first_node};
use clippy_utils::source::{SpanRangeExt, snippet_with_context};
use clippy_utils::sugg::Sugg;
use rustc_ast::{FormatArgsPiece, FormatOptions, FormatTrait};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

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
    /// ```no_run
    /// let foo = "foo";
    /// format!("{}", foo);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let foo = "foo";
    /// foo.to_owned();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_FORMAT,
    complexity,
    "useless use of `format!`"
}

#[allow(clippy::module_name_repetitions)]
pub struct UselessFormat {
    format_args: FormatArgsStorage,
}

impl UselessFormat {
    pub fn new(format_args: FormatArgsStorage) -> Self {
        Self { format_args }
    }
}

impl_lint_pass!(UselessFormat => [USELESS_FORMAT]);

impl<'tcx> LateLintPass<'tcx> for UselessFormat {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(macro_call) = root_macro_call_first_node(cx, expr)
            && cx.tcx.is_diagnostic_item(sym::format_macro, macro_call.def_id)
            && let Some(format_args) = self.format_args.get(cx, expr, macro_call.expn)
        {
            let mut applicability = Applicability::MachineApplicable;
            let call_site = macro_call.span;

            match (format_args.arguments.all_args(), &format_args.template[..]) {
                ([], []) => span_useless_format_empty(cx, call_site, "String::new()".to_owned(), applicability),
                ([], [_]) => {
                    // Simulate macro expansion, converting {{ and }} to { and }.
                    let Some(snippet) = format_args.span.get_source_text(cx) else {
                        return;
                    };
                    let s_expand = snippet.replace("{{", "{").replace("}}", "}");
                    let sugg = format!("{s_expand}.to_string()");
                    span_useless_format(cx, call_site, sugg, applicability);
                },
                ([arg], [piece]) => {
                    if let Some(value) = find_format_arg_expr(expr, arg)
                        && let FormatArgsPiece::Placeholder(placeholder) = piece
                        && placeholder.format_trait == FormatTrait::Display
                        && placeholder.format_options == FormatOptions::default()
                        && match cx.typeck_results().expr_ty(value).peel_refs().kind() {
                            ty::Adt(adt, _) => Some(adt.did()) == cx.tcx.lang_items().string(),
                            ty::Str => true,
                            _ => false,
                        }
                    {
                        let is_new_string = match value.kind {
                            ExprKind::Binary(..) => true,
                            ExprKind::MethodCall(path, ..) => path.ident.name == sym::to_string,
                            _ => false,
                        };
                        let sugg = if is_new_string {
                            snippet_with_context(cx, value.span, call_site.ctxt(), "..", &mut applicability)
                                .0
                                .into_owned()
                        } else {
                            let sugg = Sugg::hir_with_context(cx, value, call_site.ctxt(), "<arg>", &mut applicability);
                            format!("{}.to_string()", sugg.maybe_par())
                        };
                        span_useless_format(cx, call_site, sugg, applicability);
                    }
                },
                _ => {},
            }
        }
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
