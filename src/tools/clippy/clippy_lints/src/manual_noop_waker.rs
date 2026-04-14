use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::{is_empty_block, sym};
use rustc_hir::{ImplItemKind, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of `std::task::Wake` that are empty.
    ///
    /// ### Why is this bad?
    /// `Waker::noop()` provides a more performant and cleaner way to create a
    /// waker that does nothing, avoiding unnecessary `Arc` allocations and
    /// reference count increments.
    ///
    /// ### Example
    /// ```rust
    /// # use std::sync::Arc;
    /// # use std::task::Wake;
    /// struct MyWaker;
    /// impl Wake for MyWaker {
    ///     fn wake(self: Arc<Self>) {}
    ///     fn wake_by_ref(self: &Arc<Self>) {}
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// use std::task::Waker;
    /// let waker = Waker::noop();
    /// ```
    #[clippy::version = "1.96.0"]
    pub MANUAL_NOOP_WAKER,
    complexity,
    "manual implementations of noop wakers can be simplified using Waker::noop()"
}

impl_lint_pass!(ManualNoopWaker => [MANUAL_NOOP_WAKER]);

pub struct ManualNoopWaker {
    msrv: Msrv,
}

impl ManualNoopWaker {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualNoopWaker {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Impl(imp) = item.kind
            && let Some(trait_ref) = imp.of_trait
            && let Some(trait_id) = trait_ref.trait_ref.trait_def_id()
            && cx.tcx.is_diagnostic_item(sym::Wake, trait_id)
            && self.msrv.meets(cx, msrvs::WAKER_NOOP)
        {
            for impl_item_ref in imp.items {
                let impl_item = cx
                    .tcx
                    .hir_node_by_def_id(impl_item_ref.owner_id.def_id)
                    .expect_impl_item();

                if let ImplItemKind::Fn(_, body_id) = &impl_item.kind {
                    let body = cx.tcx.hir_body(*body_id);
                    if !is_empty_block(body.value) {
                        return;
                    }
                }
            }

            span_lint_and_help(
                cx,
                MANUAL_NOOP_WAKER,
                trait_ref.trait_ref.path.span,
                "manual implementation of a no-op waker",
                None,
                "use `std::task::Waker::noop()` instead",
            );
        }
    }
}
