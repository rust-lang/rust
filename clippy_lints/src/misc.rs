use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::sugg::Sugg;
use clippy_utils::{fulfill_or_allowed, in_automatically_derived};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{BinOpKind, Expr, ExprKind, HirId, Node, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::{Span, Symbol};

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

declare_lint_pass!(LintPass => [
    SHORT_CIRCUIT_STATEMENT,
    USED_UNDERSCORE_BINDING,
    USED_UNDERSCORE_ITEMS,
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
        check_used_underscore(cx, expr);
    }
}

#[derive(Clone, Copy)]
enum Id<'tcx> {
    /// A local binding.
    Binding(HirId),
    /// A resolved local definition.
    LocalDef(LocalDefId),
    /// An unresolved type-relative definition.
    TyRel,
    /// A field of an unresolved type.
    FieldOf(&'tcx Expr<'tcx>),
}
impl Id<'_> {
    fn from_res(res: Res) -> Option<Self> {
        match res {
            Res::Local(id) => Some(Self::Binding(id)),
            Res::Def(_, id) => id.as_local().map(Self::LocalDef),
            _ => None,
        }
    }

    fn get_local_def(self, cx: &LateContext<'_>, e: &Expr<'_>, name: Symbol) -> Option<(HirId, Span)> {
        let id = match self {
            Self::Binding(id) if let Node::Pat(p) = cx.tcx.hir_node(id) => return Some((id, p.span)),
            Self::LocalDef(id) => id,
            Self::TyRel => cx.typeck_results().type_dependent_def_id(e.hir_id)?.as_local()?,
            Self::FieldOf(e)
                if let ty::Adt(adt, _) = *cx.typeck_results().expr_ty_adjusted(e).kind()
                    && adt.did().is_local()
                    && let [variant] = &adt.variants().raw
                    && let Some(f) = variant.fields.iter().find(|&f| f.name == name)
                    && match *cx.tcx.type_of(f.did).instantiate_identity().skip_normalization().kind() {
                        ty::Adt(adt, _) => !adt.is_phantom_data(),
                        _ => true,
                    }
                    && let Some(id) = f.did.as_local() =>
            {
                id
            },
            Self::FieldOf(_) | Self::Binding(_) => return None,
        };
        if cx.tcx.is_foreign_item(id) {
            None
        } else {
            Some((cx.tcx.local_def_id_to_hir_id(id), cx.tcx.def_span(id)))
        }
    }
}

fn check_used_underscore<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
    let (ident, id) = match e.kind {
        ExprKind::Path(path) => match path {
            QPath::Resolved(_, path)
                if let Some(id) = Id::from_res(path.res)
                    && let [.., seg] = path.segments =>
            {
                (seg.ident, id)
            },
            QPath::Resolved(..) => return,
            QPath::TypeRelative(_, seg) => (seg.ident, Id::TyRel),
        },
        ExprKind::MethodCall(path, ..) => (path.ident, Id::TyRel),
        ExprKind::Struct(path, ..) => match path {
            QPath::Resolved(_, path)
                if let Some(id) = Id::from_res(path.res)
                    && let [.., seg] = path.segments =>
            {
                (seg.ident, id)
            },
            QPath::Resolved(..) => return,
            QPath::TypeRelative(_, seg) => (seg.ident, Id::TyRel),
        },
        ExprKind::Field(base, ident) => (ident, Id::FieldOf(base)),
        _ => return,
    };

    if !e.span.from_expansion()
        && !ident.span.from_expansion()
        && ident
            .name
            .as_str()
            .strip_prefix('_')
            .is_some_and(|x| !x.starts_with('_'))
        && let Some((def_hir_id, def_sp)) = id.get_local_def(cx, e, ident.name)
        && !def_sp.from_expansion()
        // Only lint when rustc's `unused_variables` would trigger
        && (!matches!(id, Id::Binding(_)) || is_used(cx, e))
        && !in_automatically_derived(cx.tcx, e.hir_id)
        && let (lint, msg, help_msg) = match id {
            Id::Binding(_) | Id::FieldOf(_) => (
                USED_UNDERSCORE_BINDING,
                "used underscore-prefixed binding",
                "binding is defined here",
            ),
            Id::TyRel | Id::LocalDef(_) => (
                USED_UNDERSCORE_ITEMS,
                "used underscore-prefixed item",
                "item is defined here",
            ),
        }
        && !fulfill_or_allowed(cx, lint, [def_hir_id])
    {
        span_lint_and_then(cx, lint, e.span, msg, |diag| {
            diag.span_note(def_sp, help_msg);
        });
    }
}

fn is_used(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let mut child = expr.hir_id;
    let typeck = cx.typeck_results();
    for (id, node) in cx.tcx.hir_parent_iter(child) {
        match node {
            Node::Expr(e) => match e.kind {
                ExprKind::Field(base, ..) if typeck.expr_adjustments(base).is_empty() => child = id,
                ExprKind::Assign(lhs, ..) if child == lhs.hir_id => return false,
                // Only primitive types are considered unused for compound assignment.
                // Everything else uses takes a mutable borrow of the lhs.
                ExprKind::AssignOp(_, lhs, _)
                    if child == lhs.hir_id
                        && let ty::Int(_) | ty::Uint(_) | ty::Float(_) = *typeck.node_type(child).kind() =>
                {
                    return false;
                },
                _ => return true,
            },
            _ => return true,
        }
    }
    true
}
