use crate::utils::{higher, is_direct_expn_of, span_lint};
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{BorrowKind, Expr, ExprKind, MatchSource, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for function/method calls with a mutable
    /// parameter in `debug_assert!`, `debug_assert_eq!` and `debug_assert_ne!` macros.
    ///
    /// **Why is this bad?** In release builds `debug_assert!` macros are optimized out by the
    /// compiler.
    /// Therefore mutating something in a `debug_assert!` macro results in different behaviour
    /// between a release and debug build.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust,ignore
    /// debug_assert_eq!(vec![3].pop(), Some(3));
    /// // or
    /// fn take_a_mut_parameter(_: &mut u32) -> bool { unimplemented!() }
    /// debug_assert!(take_a_mut_parameter(&mut 5));
    /// ```
    pub DEBUG_ASSERT_WITH_MUT_CALL,
    nursery,
    "mutable arguments in `debug_assert{,_ne,_eq}!`"
}

declare_lint_pass!(DebugAssertWithMutCall => [DEBUG_ASSERT_WITH_MUT_CALL]);

const DEBUG_MACRO_NAMES: [&str; 3] = ["debug_assert", "debug_assert_eq", "debug_assert_ne"];

impl<'tcx> LateLintPass<'tcx> for DebugAssertWithMutCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        for dmn in &DEBUG_MACRO_NAMES {
            if is_direct_expn_of(e.span, dmn).is_some() {
                if let Some(macro_args) = higher::extract_assert_macro_args(e) {
                    for arg in macro_args {
                        let mut visitor = MutArgVisitor::new(cx);
                        visitor.visit_expr(arg);
                        if let Some(span) = visitor.expr_span() {
                            span_lint(
                                cx,
                                DEBUG_ASSERT_WITH_MUT_CALL,
                                span,
                                &format!("do not call a function with mutable arguments inside of `{}!`", dmn),
                            );
                        }
                    }
                }
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
        if self.found {
            self.expr_span
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for MutArgVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

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
        walk_expr(self, expr)
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}
