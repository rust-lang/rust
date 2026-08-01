use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef as _, MaybeQPath as _};
use clippy_utils::source::snippet_with_context;
use clippy_utils::sym;
use clippy_utils::visitors::is_expr_unsafe;
use rustc_errors::Applicability;
use rustc_hir::{Block, BlockCheckMode, Expr, ExprKind, LangItem, Node, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unsafe usage of `NonNull::new_unchecked(Box::into_raw(x))`, and suggests calling `NonNull::from_mut(Box::leak(x))` instead.
    ///
    /// ### Why is this bad?
    /// `NonNull::new_unchecked` is an unsafe function, which we don't need to call at all if we can instead use a mutable reference.
    ///
    /// ### Example
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let one = Box::new(1);
    /// let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(one)) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::ptr::NonNull;
    /// let one = Box::new(1);
    /// let ptr = NonNull::from_mut(Box::leak(one));
    /// ```
    #[clippy::version = "1.98.0"]
    pub NONNULL_UNCHECKED_ON_BOX_PTR,
    complexity,
    "using `NonNull::new_unchecked` with `Box::into_raw`, while `NonNull::from_mut` with `Box::leak` can be used instead"
}

impl_lint_pass!(NonnullUncheckedOnBoxPtr => [NONNULL_UNCHECKED_ON_BOX_PTR]);

pub struct NonnullUncheckedOnBoxPtr {
    msrv: Msrv,
}

impl NonnullUncheckedOnBoxPtr {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonnullUncheckedOnBoxPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let ExprKind::Call(nonnull_new_unchecked, [arg]) = expr.kind
            && let ExprKind::Call(box_into_raw, [arg]) = arg.kind
            && nonnull_new_unchecked
                .ty_rel_def_if_named(cx, sym::new_unchecked)
                .opt_parent(cx)
                .opt_impl_ty(cx)
                .is_diag_item(cx, sym::NonNull)
            && box_into_raw
                .ty_rel_def_if_named(cx, sym::into_raw)
                .opt_parent(cx)
                .opt_impl_ty(cx)
                .is_lang_item(cx, LangItem::OwnedBox)
            && self.msrv.meets(cx, msrvs::BOX_LEAK)
        {
            let ctxt = expr.span.ctxt();
            let span = match cx.tcx.parent_hir_node(expr.hir_id) {
                Node::Block(&Block {
                    rules: BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                    span: unsafe_span,
                    stmts,
                    ..
                }) if unsafe_span.ctxt() == ctxt && !is_expr_unsafe(cx, arg) && stmts.is_empty() => unsafe_span,
                _ => expr.span,
            };

            span_lint_and_then(
                cx,
                NONNULL_UNCHECKED_ON_BOX_PTR,
                span,
                "use of `NonNull::new_unchecked` with `Box::into_raw`",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let arg_name = snippet_with_context(cx, arg.span, ctxt, "_", &mut app).0;

                    let sugg = if self.msrv.meets(cx, msrvs::NONNULL_FROM_MUT) {
                        format!("NonNull::from_mut(Box::leak({arg_name}))")
                    } else {
                        format!("NonNull::from(Box::leak({arg_name}))")
                    };

                    diag.span_suggestion(span, "try", sugg, app);
                },
            );
        }
    }
}
