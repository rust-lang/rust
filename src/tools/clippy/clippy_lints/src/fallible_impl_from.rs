use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_expn_of, match_panic_def_id, method_chain_args};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for impls of `From<..>` that contain `panic!()` or `unwrap()`
    ///
    /// ### Why is this bad?
    /// `TryFrom` should be used if there's a possibility of failure.
    ///
    /// ### Example
    /// ```rust
    /// struct Foo(i32);
    ///
    /// // Bad
    /// impl From<String> for Foo {
    ///     fn from(s: String) -> Self {
    ///         Foo(s.parse().unwrap())
    ///     }
    /// }
    /// ```
    ///
    /// ```rust
    /// // Good
    /// struct Foo(i32);
    ///
    /// use std::convert::TryFrom;
    /// impl TryFrom<String> for Foo {
    ///     type Error = ();
    ///     fn try_from(s: String) -> Result<Self, Self::Error> {
    ///         if let Ok(parsed) = s.parse() {
    ///             Ok(Foo(parsed))
    ///         } else {
    ///             Err(())
    ///         }
    ///     }
    /// }
    /// ```
    pub FALLIBLE_IMPL_FROM,
    nursery,
    "Warn on impls of `From<..>` that contain `panic!()` or `unwrap()`"
}

declare_lint_pass!(FallibleImplFrom => [FALLIBLE_IMPL_FROM]);

impl<'tcx> LateLintPass<'tcx> for FallibleImplFrom {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        // check for `impl From<???> for ..`
        if_chain! {
            if let hir::ItemKind::Impl(impl_) = &item.kind;
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(item.def_id);
            if cx.tcx.is_diagnostic_item(sym::From, impl_trait_ref.def_id);
            then {
                lint_impl_body(cx, item.span, impl_.items);
            }
        }
    }
}

fn lint_impl_body<'tcx>(cx: &LateContext<'tcx>, impl_span: Span, impl_items: &[hir::ImplItemRef]) {
    use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
    use rustc_hir::{Expr, ExprKind, ImplItemKind, QPath};

    struct FindPanicUnwrap<'a, 'tcx> {
        lcx: &'a LateContext<'tcx>,
        typeck_results: &'tcx ty::TypeckResults<'tcx>,
        result: Vec<Span>,
    }

    impl<'a, 'tcx> Visitor<'tcx> for FindPanicUnwrap<'a, 'tcx> {
        type Map = Map<'tcx>;

        fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
            // check for `begin_panic`
            if_chain! {
                if let ExprKind::Call(func_expr, _) = expr.kind;
                if let ExprKind::Path(QPath::Resolved(_, path)) = func_expr.kind;
                if let Some(path_def_id) = path.res.opt_def_id();
                if match_panic_def_id(self.lcx, path_def_id);
                if is_expn_of(expr.span, "unreachable").is_none();
                then {
                    self.result.push(expr.span);
                }
            }

            // check for `unwrap`
            if let Some(arglists) = method_chain_args(expr, &["unwrap"]) {
                let reciever_ty = self.typeck_results.expr_ty(&arglists[0][0]).peel_refs();
                if is_type_diagnostic_item(self.lcx, reciever_ty, sym::Option)
                    || is_type_diagnostic_item(self.lcx, reciever_ty, sym::Result)
                {
                    self.result.push(expr.span);
                }
            }

            // and check sub-expressions
            intravisit::walk_expr(self, expr);
        }

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }
    }

    for impl_item in impl_items {
        if_chain! {
            if impl_item.ident.name == sym::from;
            if let ImplItemKind::Fn(_, body_id) =
                cx.tcx.hir().impl_item(impl_item.id).kind;
            then {
                // check the body for `begin_panic` or `unwrap`
                let body = cx.tcx.hir().body(body_id);
                let mut fpu = FindPanicUnwrap {
                    lcx: cx,
                    typeck_results: cx.tcx.typeck(impl_item.id.def_id),
                    result: Vec::new(),
                };
                fpu.visit_expr(&body.value);

                // if we've found one, lint
                if !fpu.result.is_empty() {
                    span_lint_and_then(
                        cx,
                        FALLIBLE_IMPL_FROM,
                        impl_span,
                        "consider implementing `TryFrom` instead",
                        move |diag| {
                            diag.help(
                                "`From` is intended for infallible conversions only. \
                                Use `TryFrom` if there's a possibility for the conversion to fail");
                            diag.span_note(fpu.result, "potential failure(s)");
                        });
                }
            }
        }
    }
}
