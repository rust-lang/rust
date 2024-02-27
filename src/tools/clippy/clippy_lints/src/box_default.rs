use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::macro_backtrace;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::expr_sig;
use clippy_utils::{is_default_equivalent, path_def_id};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_ty, Visitor};
use rustc_hir::{Block, Expr, ExprKind, Local, Node, QPath, Ty, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::IsSuggestable;
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// checks for `Box::new(T::default())`, which is better written as
    /// `Box::<T>::default()`.
    ///
    /// ### Why is this bad?
    /// First, it's more complex, involving two calls instead of one.
    /// Second, `Box::default()` can be faster
    /// [in certain cases](https://nnethercote.github.io/perf-book/standard-library-types.html#box).
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
    perf,
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
            && path_def_id(cx, ty).map_or(false, |id| Some(id) == cx.tcx.lang_items().owned_box())
            // And the single argument to the call is another function call
            // This is the `T::default()` of `Box::new(T::default())`
            && let ExprKind::Call(arg_path, inner_call_args) = arg.kind
            // And we are not in a foreign crate's macro
            && !in_external_macro(cx.sess(), expr.span)
            // And the argument expression has the same context as the outer call expression
            // or that we are inside a `vec!` macro expansion
            && (expr.span.eq_ctxt(arg.span) || is_local_vec_expn(cx, arg, expr))
            // And the argument is equivalent to `Default::default()`
            && is_default_equivalent(cx, arg)
        {
            span_lint_and_sugg(
                cx,
                BOX_DEFAULT,
                expr.span,
                "`Box::new(_)` of default value",
                "try",
                if is_plain_default(cx, arg_path) || given_type(cx, expr) {
                    "Box::default()".into()
                } else if let Some(arg_ty) = cx.typeck_results().expr_ty(arg).make_suggestable(cx.tcx, true) {
                    // Check if we can copy from the source expression in the replacement.
                    // We need the call to have no argument (see `explicit_default_type`).
                    if inner_call_args.is_empty()
                        && let Some(ty) = explicit_default_type(arg_path)
                        && let Some(s) = snippet_opt(cx, ty.span)
                    {
                        format!("Box::<{s}>::default()")
                    } else {
                        // Otherwise, use the inferred type's formatting.
                        with_forced_trimmed_paths!(format!("Box::<{arg_ty}>::default()"))
                    }
                } else {
                    return;
                },
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

// Checks whether the call is of the form `A::B::f()`. Returns `A::B` if it is.
//
// In the event we have this kind of construct, it's easy to use `A::B` as a replacement in the
// quickfix. `f` must however have no parameter. Should `f` have some, then some of the type of
// `A::B` may be inferred from the arguments. This would be the case for `Vec::from([0; false])`,
// where the argument to `from` allows inferring this is a `Vec<bool>`
fn explicit_default_type<'a>(arg_path: &'a Expr<'_>) -> Option<&'a Ty<'a>> {
    if let ExprKind::Path(QPath::TypeRelative(ty, _)) = &arg_path.kind {
        Some(ty)
    } else {
        None
    }
}

fn is_local_vec_expn(cx: &LateContext<'_>, expr: &Expr<'_>, ref_expr: &Expr<'_>) -> bool {
    macro_backtrace(expr.span).next().map_or(false, |call| {
        cx.tcx.is_diagnostic_item(sym::vec_macro, call.def_id) && call.span.eq_ctxt(ref_expr.span)
    })
}

#[derive(Default)]
struct InferVisitor(bool);

impl<'tcx> Visitor<'tcx> for InferVisitor {
    fn visit_ty(&mut self, t: &rustc_hir::Ty<'_>) {
        self.0 |= matches!(t.kind, TyKind::Infer | TyKind::OpaqueDef(..) | TyKind::TraitObject(..));
        if !self.0 {
            walk_ty(self, t);
        }
    }
}

fn given_type(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match cx.tcx.parent_hir_node(expr.hir_id) {
        Node::Local(Local { ty: Some(ty), .. }) => {
            let mut v = InferVisitor::default();
            v.visit_ty(ty);
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
                && !cx.typeck_results().expr_ty_adjusted(expr).boxed_ty().is_trait()
            {
                input.no_bound_vars().is_some()
            } else {
                false
            }
        },
        _ => false,
    }
}
