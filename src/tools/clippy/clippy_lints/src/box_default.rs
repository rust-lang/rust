use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::macro_backtrace;
use clippy_utils::ty::expr_sig;
use clippy_utils::{is_default_equivalent, path_def_id};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{InferKind, Visitor, VisitorExt, walk_ty};
use rustc_hir::{AmbigArg, Block, Expr, ExprKind, HirId, LetStmt, Node, QPath, Ty, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::{Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// checks for `Box::new(Default::default())`, which can be written as
    /// `Box::default()`.
    ///
    /// ### Why is this bad?
    /// `Box::default()` is equivalent and more concise.
    ///
    /// ### Example
    /// ```no_run
    /// let x: Box<String> = Box::new(Default::default());
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: Box<String> = Box::default();
    /// ```
    #[clippy::version = "1.66.0"]
    pub BOX_DEFAULT,
    style,
    "Using Box::new(T::default()) instead of Box::default()"
}

declare_lint_pass!(BoxDefault => [BOX_DEFAULT]);

impl LateLintPass<'_> for BoxDefault {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        // If the expression is a call (`Box::new(...)`)
        if let ExprKind::Call(box_new, [arg]) = expr.kind
            // And call is of the form `<T>::something`
            // Here, it would be `<Box>::new`
            && let ExprKind::Path(QPath::TypeRelative(ty, seg)) = box_new.kind
            // And that method is `new`
            && seg.ident.name == sym::new
            // And the call is that of a `Box` method
            && path_def_id(cx, ty).is_some_and(|id| Some(id) == cx.tcx.lang_items().owned_box())
            // And the single argument to the call is another function call
            // This is the `T::default()` (or default equivalent) of `Box::new(T::default())`
            && let ExprKind::Call(arg_path, _) = arg.kind
            // And we are not in a foreign crate's macro
            && !expr.span.in_external_macro(cx.sess().source_map())
            // And the argument expression has the same context as the outer call expression
            // or that we are inside a `vec!` macro expansion
            && (expr.span.eq_ctxt(arg.span) || is_local_vec_expn(cx, arg, expr))
            // And the argument is `Default::default()` or the type is specified
            && (is_plain_default(cx, arg_path) || (given_type(cx, expr) && is_default_equivalent(cx, arg)))
        {
            span_lint_and_sugg(
                cx,
                BOX_DEFAULT,
                expr.span,
                "`Box::new(_)` of default value",
                "try",
                "Box::default()".into(),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_plain_default(cx: &LateContext<'_>, arg_path: &Expr<'_>) -> bool {
    // we need to match the actual path so we don't match e.g. "u8::default"
    if let ExprKind::Path(QPath::Resolved(None, path)) = &arg_path.kind
        && let Res::Def(_, def_id) = path.res
    {
        // avoid generic parameters
        cx.tcx.is_diagnostic_item(sym::default_fn, def_id) && path.segments.iter().all(|seg| seg.args.is_none())
    } else {
        false
    }
}

fn is_local_vec_expn(cx: &LateContext<'_>, expr: &Expr<'_>, ref_expr: &Expr<'_>) -> bool {
    macro_backtrace(expr.span)
        .next()
        .is_some_and(|call| cx.tcx.is_diagnostic_item(sym::vec_macro, call.def_id) && call.span.eq_ctxt(ref_expr.span))
}

#[derive(Default)]
struct InferVisitor(bool);

impl Visitor<'_> for InferVisitor {
    fn visit_infer(&mut self, inf_id: HirId, _inf_span: Span, _kind: InferKind<'_>) -> Self::Result {
        self.0 = true;
        self.visit_id(inf_id);
    }

    fn visit_ty(&mut self, t: &Ty<'_, AmbigArg>) {
        self.0 |= matches!(t.kind, TyKind::OpaqueDef(..) | TyKind::TraitObject(..));
        if !self.0 {
            walk_ty(self, t);
        }
    }
}

fn given_type(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match cx.tcx.parent_hir_node(expr.hir_id) {
        Node::LetStmt(LetStmt { ty: Some(ty), .. }) => {
            let mut v = InferVisitor::default();
            v.visit_ty_unambig(ty);
            !v.0
        },
        Node::Expr(Expr {
            kind: ExprKind::Call(path, args),
            ..
        })
        | Node::Block(Block {
            expr: Some(Expr {
                kind: ExprKind::Call(path, args),
                ..
            }),
            ..
        }) => {
            if let Some(index) = args.iter().position(|arg| arg.hir_id == expr.hir_id)
                && let Some(sig) = expr_sig(cx, path)
                && let Some(input) = sig.input(index)
                && let Some(input_ty) = input.no_bound_vars()
            {
                input_ty == cx.typeck_results().expr_ty_adjusted(expr)
            } else {
                false
            }
        },
        _ => false,
    }
}
