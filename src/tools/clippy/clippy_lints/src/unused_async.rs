use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::is_def_id_trait_method;
use rustc_hir::def::DefKind;
use rustc_hir::intravisit::{walk_expr, walk_fn, FnKind, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnDecl, Node, YieldSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_session::impl_lint_pass;
use rustc_span::def_id::{LocalDefId, LocalDefIdSet};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that are declared `async` but have no `.await`s inside of them.
    ///
    /// ### Why is this bad?
    /// Async functions with no async code create overhead, both mentally and computationally.
    /// Callers of async methods either need to be calling from an async function themselves or run it on an executor, both of which
    /// causes runtime overhead and hassle for the caller.
    ///
    /// ### Example
    /// ```no_run
    /// async fn get_random_number() -> i64 {
    ///     4 // Chosen by fair dice roll. Guaranteed to be random.
    /// }
    /// let number_future = get_random_number();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn get_random_number_improved() -> i64 {
    ///     4 // Chosen by fair dice roll. Guaranteed to be random.
    /// }
    /// let number_future = async { get_random_number_improved() };
    /// ```
    #[clippy::version = "1.54.0"]
    pub UNUSED_ASYNC,
    pedantic,
    "finds async functions with no await statements"
}

#[derive(Default)]
pub struct UnusedAsync {
    /// Keeps track of async functions used as values (i.e. path expressions to async functions that
    /// are not immediately called)
    async_fns_as_value: LocalDefIdSet,
    /// Functions with unused `async`, linted post-crate after we've found all uses of local async
    /// functions
    unused_async_fns: Vec<UnusedAsyncFn>,
}

#[derive(Copy, Clone)]
struct UnusedAsyncFn {
    def_id: LocalDefId,
    fn_span: Span,
    await_in_async_block: Option<Span>,
}

impl_lint_pass!(UnusedAsync => [UNUSED_ASYNC]);

struct AsyncFnVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    found_await: bool,
    /// Also keep track of `await`s in nested async blocks so we can mention
    /// it in a note
    await_in_async_block: Option<Span>,
    async_depth: usize,
}

impl<'a, 'tcx> Visitor<'tcx> for AsyncFnVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if let ExprKind::Yield(_, YieldSource::Await { .. }) = ex.kind {
            if self.async_depth == 1 {
                self.found_await = true;
            } else if self.await_in_async_block.is_none() {
                self.await_in_async_block = Some(ex.span);
            }
        }

        let is_async_block = matches!(
            ex.kind,
            ExprKind::Closure(rustc_hir::Closure {
                kind: rustc_hir::ClosureKind::Coroutine(rustc_hir::CoroutineKind::Desugared(
                    rustc_hir::CoroutineDesugaring::Async,
                    _
                )),
                ..
            })
        );

        if is_async_block {
            self.async_depth += 1;
        }

        walk_expr(self, ex);

        if is_async_block {
            self.async_depth -= 1;
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

impl<'tcx> LateLintPass<'tcx> for UnusedAsync {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'tcx>,
        body: &Body<'tcx>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if !span.from_expansion() && fn_kind.asyncness().is_async() && !is_def_id_trait_method(cx, def_id) {
            let mut visitor = AsyncFnVisitor {
                cx,
                found_await: false,
                async_depth: 0,
                await_in_async_block: None,
            };
            walk_fn(&mut visitor, fn_kind, fn_decl, body.id(), def_id);
            if !visitor.found_await {
                // Don't lint just yet, but store the necessary information for later.
                // The actual linting happens in `check_crate_post`, once we've found all
                // uses of local async functions that do require asyncness to pass typeck
                self.unused_async_fns.push(UnusedAsyncFn {
                    await_in_async_block: visitor.await_in_async_block,
                    fn_span: span,
                    def_id,
                });
            }
        }
    }

    fn check_path(&mut self, cx: &LateContext<'tcx>, path: &rustc_hir::Path<'tcx>, hir_id: rustc_hir::HirId) {
        fn is_node_func_call(node: Node<'_>, expected_receiver: Span) -> bool {
            matches!(
                node,
                Node::Expr(Expr {
                    kind: ExprKind::Call(Expr { span, .. }, _) | ExprKind::MethodCall(_, Expr { span, .. }, ..),
                    ..
                }) if *span == expected_receiver
            )
        }

        // Find paths to local async functions that aren't immediately called.
        // E.g. `async fn f() {}; let x = f;`
        // Depending on how `x` is used, f's asyncness might be required despite not having any `await`
        // statements, so don't lint at all if there are any such paths.
        if let Some(def_id) = path.res.opt_def_id()
            && let Some(local_def_id) = def_id.as_local()
            && cx.tcx.def_kind(def_id) == DefKind::Fn
            && cx.tcx.asyncness(def_id).is_async()
            && !is_node_func_call(cx.tcx.parent_hir_node(hir_id), path.span)
        {
            self.async_fns_as_value.insert(local_def_id);
        }
    }

    // After collecting all unused `async` and problematic paths to such functions,
    // lint those unused ones that didn't have any path expressions to them.
    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let iter = self
            .unused_async_fns
            .iter()
            .filter(|UnusedAsyncFn { def_id, .. }| (!self.async_fns_as_value.contains(def_id)));

        for fun in iter {
            span_lint_hir_and_then(
                cx,
                UNUSED_ASYNC,
                cx.tcx.local_def_id_to_hir_id(fun.def_id),
                fun.fn_span,
                "unused `async` for function with no await statements",
                |diag| {
                    diag.help("consider removing the `async` from this function");

                    if let Some(span) = fun.await_in_async_block {
                        diag.span_note(
                            span,
                            "`await` used in an async block, which does not require \
                            the enclosing function to be `async`",
                        );
                    }
                },
            );
        }
    }
}
