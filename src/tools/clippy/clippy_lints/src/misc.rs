use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::sugg::Sugg;
use clippy_utils::{SpanlessEq, fulfill_or_allowed, get_parent_expr, in_automatically_derived, last_path_segment};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of bindings with a single leading
    /// underscore.
    ///
    /// ### Why is this bad?
    /// A single leading underscore is usually used to indicate
    /// that a binding will not be used. Using such a binding breaks this
    /// expectation.
    ///
    /// ### Known problems
    /// The lint does not work properly with desugaring and
    /// macro, it has been allowed in the meantime.
    ///
    /// ### Example
    /// ```no_run
    /// let _x = 0;
    /// let y = _x + 1; // Here we are using `_x`, even though it has a leading
    ///                 // underscore. We should rename `_x` to `x`
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USED_UNDERSCORE_BINDING,
    pedantic,
    "using a binding which is prefixed with an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of item with a single leading
    /// underscore.
    ///
    /// ### Why is this bad?
    /// A single leading underscore is usually used to indicate
    /// that a item will not be used. Using such a item breaks this
    /// expectation.
    ///
    /// ### Example
    /// ```no_run
    /// fn _foo() {}
    ///
    /// struct _FooStruct {}
    ///
    /// fn main() {
    ///     _foo();
    ///     let _ = _FooStruct{};
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo() {}
    ///
    /// struct FooStruct {}
    ///
    /// fn main() {
    ///     foo();
    ///     let _ = FooStruct{};
    /// }
    /// ```
    #[clippy::version = "1.83.0"]
    pub USED_UNDERSCORE_ITEMS,
    pedantic,
    "using a item which is prefixed with an underscore"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for the use of short circuit boolean conditions as
    /// a
    /// statement.
    ///
    /// ### Why is this bad?
    /// Using a short circuit boolean condition as a statement
    /// may hide the fact that the second part is executed or not depending on the
    /// outcome of the first part.
    ///
    /// ### Example
    /// ```rust,ignore
    /// f() && g(); // We should write `if f() { g(); }`.
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SHORT_CIRCUIT_STATEMENT,
    complexity,
    "using a short circuit boolean condition as a statement"
}

declare_lint_pass!(LintPass => [
    USED_UNDERSCORE_BINDING,
    USED_UNDERSCORE_ITEMS,
    SHORT_CIRCUIT_STATEMENT,
]);

impl<'tcx> LateLintPass<'tcx> for LintPass {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Semi(expr) = stmt.kind
            && let ExprKind::Binary(binop, a, b) = &expr.kind
            && matches!(binop.node, BinOpKind::And | BinOpKind::Or)
            && !stmt.span.from_expansion()
            && expr.span.eq_ctxt(stmt.span)
        {
            span_lint_hir_and_then(
                cx,
                SHORT_CIRCUIT_STATEMENT,
                expr.hir_id,
                stmt.span,
                "boolean short circuit operator in statement may be clearer using an explicit test",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let test = Sugg::hir_with_context(cx, a, expr.span.ctxt(), "_", &mut app);
                    let test = if binop.node == BinOpKind::Or { !test } else { test };
                    let then = Sugg::hir_with_context(cx, b, expr.span.ctxt(), "_", &mut app);
                    diag.span_suggestion(stmt.span, "replace it with", format!("if {test} {{ {then}; }}"), app);
                },
            );
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.in_external_macro(cx.sess().source_map())
            || expr.span.desugaring_kind().is_some()
            || in_automatically_derived(cx.tcx, expr.hir_id)
        {
            return;
        }

        used_underscore_binding(cx, expr);
        used_underscore_items(cx, expr);
    }
}

fn used_underscore_items<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    let (def_id, ident) = match expr.kind {
        ExprKind::Call(func, ..) => {
            if let ExprKind::Path(QPath::Resolved(.., path)) = func.kind
                && let Some(last_segment) = path.segments.last()
                && let Res::Def(_, def_id) = last_segment.res
            {
                (def_id, last_segment.ident)
            } else {
                return;
            }
        },
        ExprKind::MethodCall(path, ..) => {
            if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) {
                (def_id, path.ident)
            } else {
                return;
            }
        },
        ExprKind::Struct(QPath::Resolved(_, path), ..) => {
            if let Some(last_segment) = path.segments.last()
                && let Res::Def(_, def_id) = last_segment.res
            {
                (def_id, last_segment.ident)
            } else {
                return;
            }
        },
        _ => return,
    };

    let name = ident.name.as_str();
    let definition_span = cx.tcx.def_span(def_id);
    if name.starts_with('_')
        && !name.starts_with("__")
        && !definition_span.from_expansion()
        && def_id.is_local()
        && !cx.tcx.is_foreign_item(def_id)
    {
        span_lint_and_then(
            cx,
            USED_UNDERSCORE_ITEMS,
            expr.span,
            "used underscore-prefixed item".to_string(),
            |diag| {
                diag.span_note(definition_span, "item is defined here".to_string());
            },
        );
    }
}

fn used_underscore_binding<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    let (definition_hir_id, ident) = match expr.kind {
        ExprKind::Path(ref qpath) => {
            if let QPath::Resolved(None, path) = qpath
                && let Res::Local(id) = path.res
                && is_used(cx, expr)
            {
                (id, last_path_segment(qpath).ident)
            } else {
                return;
            }
        },
        ExprKind::Field(recv, ident) => {
            if let Some(adt_def) = cx.typeck_results().expr_ty_adjusted(recv).ty_adt_def()
                && let Some(field) = adt_def.all_fields().find(|field| field.name == ident.name)
                && let Some(local_did) = field.did.as_local()
                && !cx.tcx.type_of(field.did).skip_binder().is_phantom_data()
            {
                (cx.tcx.local_def_id_to_hir_id(local_did), ident)
            } else {
                return;
            }
        },
        _ => return,
    };

    let name = ident.name.as_str();
    if name.starts_with('_')
        && !name.starts_with("__")
        && let definition_span = cx.tcx.hir_span(definition_hir_id)
        && !definition_span.from_expansion()
        && !fulfill_or_allowed(cx, USED_UNDERSCORE_BINDING, [expr.hir_id, definition_hir_id])
    {
        span_lint_and_then(
            cx,
            USED_UNDERSCORE_BINDING,
            expr.span,
            "used underscore-prefixed binding".to_string(),
            |diag| {
                diag.span_note(definition_span, "binding is defined here".to_string());
            },
        );
    }
}

/// Heuristic to see if an expression is used. Should be compatible with
/// `unused_variables`'s idea
/// of what it means for an expression to be "used".
fn is_used(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    get_parent_expr(cx, expr).is_none_or(|parent| match parent.kind {
        ExprKind::Assign(_, rhs, _) | ExprKind::AssignOp(_, _, rhs) => SpanlessEq::new(cx).eq_expr(rhs, expr),
        _ => is_used(cx, parent),
    })
}
