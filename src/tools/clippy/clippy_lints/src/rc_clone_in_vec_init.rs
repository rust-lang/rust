use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::source::{indent_of, snippet};
use clippy_utils::{last_path_segment, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for reference-counted pointers (`Arc`, `Rc`, `rc::Weak`, and `sync::Weak`)
    /// in `vec![elem; len]`
    ///
    /// ### Why is this bad?
    /// This will create `elem` once and clone it `len` times - doing so with `Arc`/`Rc`/`Weak`
    /// is a bit misleading, as it will create references to the same pointer, rather
    /// than different instances.
    ///
    /// ### Example
    /// ```no_run
    /// let v = vec![std::sync::Arc::new("some data".to_string()); 100];
    /// // or
    /// let v = vec![std::rc::Rc::new("some data".to_string()); 100];
    /// ```
    /// Use instead:
    /// ```no_run
    /// // Initialize each value separately:
    /// let mut data = Vec::with_capacity(100);
    /// for _ in 0..100 {
    ///     data.push(std::rc::Rc::new("some data".to_string()));
    /// }
    ///
    /// // Or if you want clones of the same reference,
    /// // Create the reference beforehand to clarify that
    /// // it should be cloned for each value
    /// let data = std::rc::Rc::new("some data".to_string());
    /// let v = vec![data; 100];
    /// ```
    #[clippy::version = "1.63.0"]
    pub RC_CLONE_IN_VEC_INIT,
    suspicious,
    "initializing reference-counted pointer in `vec![elem; len]`"
}
declare_lint_pass!(RcCloneInVecInit => [RC_CLONE_IN_VEC_INIT]);

impl LateLintPass<'_> for RcCloneInVecInit {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else {
            return;
        };
        let Some(VecArgs::Repeat(elem, len)) = VecArgs::hir(cx, expr) else {
            return;
        };
        let Some((symbol, func_span)) = ref_init(cx, elem) else {
            return;
        };

        emit_lint(cx, symbol, macro_call.span, elem, len, func_span);
    }
}

fn loop_init_suggestion(elem: &str, len: &str, indent: &str) -> String {
    format!(
        r"{{
{indent}    let mut v = Vec::with_capacity({len});
{indent}    (0..{len}).for_each(|_| v.push({elem}));
{indent}    v
{indent}}}"
    )
}

fn extract_suggestion(elem: &str, len: &str, indent: &str) -> String {
    format!(
        "{{
{indent}    let data = {elem};
{indent}    vec![data; {len}]
{indent}}}"
    )
}

fn emit_lint(cx: &LateContext<'_>, symbol: Symbol, lint_span: Span, elem: &Expr<'_>, len: &Expr<'_>, func_span: Span) {
    let symbol_name = symbol.as_str();

    span_lint_and_then(
        cx,
        RC_CLONE_IN_VEC_INIT,
        lint_span,
        "initializing a reference-counted pointer in `vec![elem; len]`",
        |diag| {
            let len_snippet = snippet(cx, len.span, "..");
            let elem_snippet = format!("{}(..)", snippet(cx, elem.span.with_hi(func_span.hi()), ".."));
            let indentation = " ".repeat(indent_of(cx, lint_span).unwrap_or(0));
            let loop_init_suggestion = loop_init_suggestion(&elem_snippet, len_snippet.as_ref(), &indentation);
            let extract_suggestion = extract_suggestion(&elem_snippet, len_snippet.as_ref(), &indentation);

            diag.note(format!("each element will point to the same `{symbol_name}` instance"));
            diag.span_suggestion(
                lint_span,
                format!("consider initializing each `{symbol_name}` element individually"),
                loop_init_suggestion,
                Applicability::HasPlaceholders,
            );
            diag.span_suggestion(
                lint_span,
                format!(
                    "or if this is intentional, consider extracting the `{symbol_name}` initialization to a variable"
                ),
                extract_suggestion,
                Applicability::HasPlaceholders,
            );
        },
    );
}

/// Checks whether the given `expr` is a call to `Arc::new`, `Rc::new`, or evaluates to a `Weak`
fn ref_init(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<(Symbol, Span)> {
    if let ExprKind::Call(func, _args) = expr.kind
        && let ExprKind::Path(ref func_path @ QPath::TypeRelative(ty, _)) = func.kind
        && let TyKind::Path(ref ty_path) = ty.kind
        && let Some(def_id) = cx.qpath_res(ty_path, ty.hir_id).opt_def_id()
    {
        if last_path_segment(func_path).ident.name == sym::new
            && let Some(symbol) = cx
                .tcx
                .get_diagnostic_name(def_id)
                .filter(|symbol| symbol == &sym::Arc || symbol == &sym::Rc)
        {
            return Some((symbol, func.span));
        }

        if let ty::Adt(adt, _) = *cx.typeck_results().expr_ty(expr).kind()
            && matches!(cx.tcx.get_diagnostic_name(adt.did()), Some(sym::RcWeak | sym::ArcWeak))
        {
            return Some((sym::Weak, func.span));
        }
    }

    None
}
