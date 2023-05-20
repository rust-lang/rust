use clippy_utils::diagnostics::span_lint;
use clippy_utils::macros::{find_assert_eq_args, root_macro_call_first_node};
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{BorrowKind, Expr, ExprKind, MatchSource, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for function/method calls with a mutable
    /// parameter in `debug_assert!`, `debug_assert_eq!` and `debug_assert_ne!` macros.
    ///
    /// ### Why is this bad?
    /// In release builds `debug_assert!` macros are optimized out by the
    /// compiler.
    /// Therefore mutating something in a `debug_assert!` macro results in different behavior
    /// between a release and debug build.
    ///
    /// ### Example
    /// ```rust,ignore
    /// debug_assert_eq!(vec![3].pop(), Some(3));
    ///
    /// // or
    ///
    /// # let mut x = 5;
    /// # fn takes_a_mut_parameter(_: &mut u32) -> bool { unimplemented!() }
    /// debug_assert!(takes_a_mut_parameter(&mut x));
    /// ```
    #[clippy::version = "1.40.0"]
    pub DEBUG_ASSERT_WITH_MUT_CALL,
    nursery,
    "mutable arguments in `debug_assert{,_ne,_eq}!`"
}

declare_lint_pass!(DebugAssertWithMutCall => [DEBUG_ASSERT_WITH_MUT_CALL]);

impl<'tcx> LateLintPass<'tcx> for DebugAssertWithMutCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        let Some(macro_call) = root_macro_call_first_node(cx, e) else { return };
        let macro_name = cx.tcx.item_name(macro_call.def_id);
        if !matches!(
            macro_name.as_str(),
            "debug_assert" | "debug_assert_eq" | "debug_assert_ne"
        ) {
            return;
        }
        let Some((lhs, rhs, _)) = find_assert_eq_args(cx, e, macro_call.expn) else { return };
        for arg in [lhs, rhs] {
            let mut visitor = MutArgVisitor::new(cx);
            visitor.visit_expr(arg);
            if let Some(span) = visitor.expr_span() {
                span_lint(
                    cx,
                    DEBUG_ASSERT_WITH_MUT_CALL,
                    span,
                    &format!("do not call a function with mutable arguments inside of `{macro_name}!`"),
                );
            }
        }
    }
}

struct MutArgVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    expr_span: Option<Span>,
    found: bool,
}

impl<'a, 'tcx> MutArgVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            expr_span: None,
            found: false,
        }
    }

    fn expr_span(&self) -> Option<Span> {
        if self.found { self.expr_span } else { None }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MutArgVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        match expr.kind {
            ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _) => {
                self.found = true;
                return;
            },
            ExprKind::Path(_) => {
                if let Some(adj) = self.cx.typeck_results().adjustments().get(expr.hir_id) {
                    if adj
                        .iter()
                        .any(|a| matches!(a.target.kind(), ty::Ref(_, _, Mutability::Mut)))
                    {
                        self.found = true;
                        return;
                    }
                }
            },
            // Don't check await desugars
            ExprKind::Match(_, _, MatchSource::AwaitDesugar) => return,
            _ if !self.found => self.expr_span = Some(expr.span),
            _ => return,
        }
        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}
