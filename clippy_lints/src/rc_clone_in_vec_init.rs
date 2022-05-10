use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::last_path_segment;
use clippy_utils::macros::{root_macro_call_first_node, MacroCall};
use rustc_hir::{Expr, ExprKind, QPath, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Arc::new` or `Rc::new` in `vec![elem; len]`
    ///
    /// ### Why is this bad?
    /// This will create `elem` once and clone it `len` times - doing so with `Arc` or `Rc`
    /// is a bit misleading, as it will create references to the same pointer, rather
    /// than different instances.
    ///
    /// ### Example
    /// ```rust
    /// let v = vec![std::sync::Arc::new("some data".to_string()); 100];
    /// // or
    /// let v = vec![std::rc::Rc::new("some data".to_string()); 100];
    /// ```
    /// Use instead:
    /// ```rust
    ///
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
    #[clippy::version = "1.62.0"]
    pub RC_CLONE_IN_VEC_INIT,
    suspicious,
    "initializing `Arc` or `Rc` in `vec![elem; len]`"
}
declare_lint_pass!(RcCloneInVecInit => [RC_CLONE_IN_VEC_INIT]);

impl LateLintPass<'_> for RcCloneInVecInit {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, expr) else { return; };
        let Some(VecArgs::Repeat(elem, _)) = VecArgs::hir(cx, expr) else { return; };
        let Some(symbol) = new_reference_call(cx, elem) else { return; };

        emit_lint(cx, symbol, &macro_call);
    }
}

fn emit_lint(cx: &LateContext<'_>, symbol: Symbol, macro_call: &MacroCall) {
    let symbol_name = symbol.as_str();

    span_lint_and_then(
        cx,
        RC_CLONE_IN_VEC_INIT,
        macro_call.span,
        &format!("calling `{symbol_name}::new` in `vec![elem; len]`"),
        |diag| {
            diag.note(format!("each element will point to the same `{symbol_name}` instance"));
            diag.help(format!(
                "if this is intentional, consider extracting the `{symbol_name}` initialization to a variable"
            ));
            diag.help("or if not, initialize each element individually");
        },
    );
}

/// Checks whether the given `expr` is a call to `Arc::new` or `Rc::new`
fn new_reference_call(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<Symbol> {
    if_chain! {
        if let ExprKind::Call(func, _args) = expr.kind;
        if let ExprKind::Path(ref func_path @ QPath::TypeRelative(ty, _)) = func.kind;
        if let TyKind::Path(ref ty_path) = ty.kind;
        if let Some(def_id) = cx.qpath_res(ty_path, ty.hir_id).opt_def_id();
        if last_path_segment(func_path).ident.name == sym::new;

        then {
            return cx.tcx.get_diagnostic_name(def_id).filter(|symbol| symbol == &sym::Arc || symbol == &sym::Rc);
        }
    }

    None
}
