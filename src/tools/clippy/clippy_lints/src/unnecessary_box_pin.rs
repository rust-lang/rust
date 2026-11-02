use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::visitors::for_each_local_use_after_expr;
use clippy_utils::{is_from_proc_macro, std_or_core, sym};
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{Expr, ExprKind, LetStmt, Node, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext as _, impl_lint_pass};
use rustc_span::Span;
use rustc_span::symbol::kw;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `Box::pin` with the `Pin<Box<_>>` not moved and only used in places where a `Pin<&mut _>`
    /// suffices, in which case the `pin!` macro can be used.
    ///
    /// ### Why is this bad?
    /// `Box::pin` creates an extra heap allocation for the pointee, while `pin!` creates a local `Pin<&mut T>`,
    /// so this saves an extra heap allocation.
    ///
    /// See the documentation for [`pin!`](https://doc.rust-lang.org/stable/std/pin/macro.pin.html)
    /// for a more detailed explanation on how these two differ.
    ///
    /// ### Known issues
    /// Currently the lint is fairly limited and only emits a warning if the pinned box is used through `.as_mut()`
    /// to prevent false positives w.r.t. lifetimes
    /// (`Pin<Box<_>>` returned by `Box::pin` is `'static`, `Pin<&mut _>` returned by `pin!` is not).
    ///
    /// The following works with `Box::pin` but not with `pin!`:
    /// ```
    /// fn assert_static<T: 'static>(_: T) {}
    /// assert_static(Box::pin(async {}));
    /// ```
    ///
    /// Restricting to only lint `.as_mut()` means that we end up with a temporary in both cases,
    /// so if it compiled with `.as_mut()`, then it ought to work with `pin!` as well.
    ///
    /// ### Example
    /// ```no_run
    /// # #![feature(noop_waker)]
    /// # use std::task::{Poll, Waker, Context};
    /// # use std::future::Future;
    ///
    /// fn now_or_never<F: Future>(fut: F) -> Option<F::Output> {
    ///     let mut fut = Box::pin(fut);
    ///
    ///     match fut.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
    ///         Poll::Ready(val) => Some(val),
    ///         Poll::Pending => None
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # #![feature(noop_waker)]
    /// # use std::task::{Poll, Waker, Context};
    /// # use std::future::Future;
    ///
    /// fn now_or_never<F: Future>(fut: F) -> Option<F::Output> {
    ///     let mut fut = std::pin::pin!(fut);
    ///
    ///     match fut.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
    ///         Poll::Ready(val) => Some(val),
    ///         Poll::Pending => None
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.84.0"]
    pub UNNECESSARY_BOX_PIN,
    perf,
    "using `Box::pin` where `pin!` suffices"
}

impl_lint_pass!(UnnecessaryBoxPin => [UNNECESSARY_BOX_PIN]);

pub struct UnnecessaryBoxPin {
    msrv: Msrv,
}

impl UnnecessaryBoxPin {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv.into() }
    }
}

impl<'tcx> LateLintPass<'tcx> for UnnecessaryBoxPin {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(callee, [_]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(bx, segment)) = &callee.kind
            && cx.typeck_results().node_type(bx.hir_id).is_box()
            && segment.ident.name == kw::Pin
            && let Some(enclosing_body) = cx.enclosing_body
            && let Some(std_or_core) = std_or_core(cx)
            && self.msrv.meets(cx, msrvs::PIN_MACRO)
            && !expr.span.in_external_macro(cx.sess().source_map())
            && !is_from_proc_macro(cx, expr)
        {
            let enclosing_body_def_id = cx.tcx.hir_body_owner_def_id(enclosing_body);

            if let ControlFlow::Continue(as_mut_span) = check_pin_box_use(cx, expr, enclosing_body_def_id) {
                span_lint_and_then(
                    cx,
                    UNNECESSARY_BOX_PIN,
                    expr.span,
                    "pinning a value with `Box::pin` when local pinning suffices",
                    |diag| {
                        let mut replacements = vec![(callee.span, format!("{std_or_core}::pin::pin!"))];
                        replacements.extend(as_mut_span.map(|span| (span, String::new())));

                        diag.multipart_suggestion(
                            "use the `pin!` macro",
                            replacements,
                            Applicability::MachineApplicable,
                        );
                    },
                );
            }
        }
    }
}

/// Checks how a `Pin<Box<_>>` is used. Returns `Continue(span)` if this use is valid with
/// `Box::pin` changed to `pin!`.
///
/// The span is the `.as_mut()` span that can be safely removed.
/// Note that it's currently only returned if `Box::pin()` is the receiver of it (and not first
/// stored in a binding) to avoid move errors.
///
/// That is, `as_mut` can be safely removed here:
/// ```ignore
/// - Box::pin(async {}).as_mut().poll(...);
/// + pin!(async {}).poll(...);
/// ```
///
/// but not here, as the poll call consumes it and the binding cannot be used again in subsequent
/// iterations:
/// ```ignore
/// - let mut bx = Box::pin(async {});
/// + let mut bx = pin!(async {});
///   loop {
/// -     bx.as_mut().poll(...);
/// +     bx.poll(...);
///   }
/// ```
fn check_pin_box_use<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    enclosing_body: LocalDefId,
) -> ControlFlow<(), Option<Span>> {
    match cx.tcx.parent_hir_node(expr.hir_id) {
        Node::Expr(as_mut_expr)
            if let ExprKind::MethodCall(segment, recv, [], span) = as_mut_expr.kind
                && segment.ident.name == sym::as_mut =>
        {
            ControlFlow::Continue(Some(span.with_lo(recv.span.hi())))
        },
        Node::LetStmt(LetStmt { pat, ty: None, .. }) if let PatKind::Binding(_, local_id, ..) = pat.kind => {
            for_each_local_use_after_expr(cx, local_id, expr.hir_id, |expr| {
                if let Node::Expr(as_mut_expr) = cx.tcx.parent_hir_node(expr.hir_id)
                    && let ExprKind::MethodCall(segment, _, [], _) = as_mut_expr.kind
                    && segment.ident.name == sym::as_mut
                    // Make sure the as_mut call isn't in another body as changing `Pin<Box<_>>` to `Pin<&mut _>`
                    // can make a closure capturing it no longer satisfy 'static bounds
                    && cx.tcx.hir_enclosing_body_owner(expr.hir_id) == enclosing_body
                {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            })?;
            ControlFlow::Continue(None)
        },
        _ => ControlFlow::Break(()),
    }
}
