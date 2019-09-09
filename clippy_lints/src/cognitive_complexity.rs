//! calculate cognitive complexity and warn about overly complex functions

use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use syntax::ast::Attribute;
use syntax::source_map::Span;

use crate::utils::{match_type, paths, span_help_and_lint, LimitStack};

declare_clippy_lint! {
    /// **What it does:** Checks for methods with high cognitive complexity.
    ///
    /// **Why is this bad?** Methods of high cognitive complexity tend to be hard to
    /// both read and maintain. Also LLVM will tend to optimize small methods better.
    ///
    /// **Known problems:** Sometimes it's hard to find a way to reduce the
    /// complexity.
    ///
    /// **Example:** No. You'll see it when you get the warning.
    pub COGNITIVE_COMPLEXITY,
    complexity,
    "functions that should be split up into multiple functions"
}

pub struct CognitiveComplexity {
    limit: LimitStack,
}

impl CognitiveComplexity {
    pub fn new(limit: u64) -> Self {
        Self {
            limit: LimitStack::new(limit),
        }
    }
}

impl_lint_pass!(CognitiveComplexity => [COGNITIVE_COMPLEXITY]);

impl CognitiveComplexity {
    fn check<'a, 'tcx>(&mut self, cx: &'a LateContext<'a, 'tcx>, body: &'tcx Body, span: Span) {
        if span.from_expansion() {
            return;
        }

        let expr = &body.value;

        let mut helper = CCHelper { cc: 1, returns: 0 };
        helper.visit_expr(expr);
        let CCHelper { cc, returns } = helper;
        let ret_ty = cx.tables.node_type(expr.hir_id);
        let ret_adjust = if match_type(cx, ret_ty, &paths::RESULT) {
            returns
        } else {
            #[allow(clippy::integer_division)]
            (returns / 2)
        };

        let mut rust_cc = cc;
        // prevent degenerate cases where unreachable code contains `return` statements
        if rust_cc >= ret_adjust {
            rust_cc -= ret_adjust;
        }
        if rust_cc > self.limit.limit() {
            span_help_and_lint(
                cx,
                COGNITIVE_COMPLEXITY,
                span,
                &format!(
                    "the function has a cognitive complexity of ({}/{})",
                    rust_cc,
                    self.limit.limit()
                ),
                "you could split it up into multiple smaller functions",
            );
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CognitiveComplexity {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        hir_id: HirId,
    ) {
        let def_id = cx.tcx.hir().local_def_id(hir_id);
        if !cx.tcx.has_attr(def_id, sym!(test)) {
            self.check(cx, body, span);
        }
    }

    fn enter_lint_attrs(&mut self, cx: &LateContext<'a, 'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.push_attrs(cx.sess(), attrs, "cognitive_complexity");
    }
    fn exit_lint_attrs(&mut self, cx: &LateContext<'a, 'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.pop_attrs(cx.sess(), attrs, "cognitive_complexity");
    }
}

struct CCHelper {
    cc: u64,
    returns: u64,
}

impl<'tcx> Visitor<'tcx> for CCHelper {
    fn visit_expr(&mut self, e: &'tcx Expr) {
        walk_expr(self, e);
        match e.node {
            ExprKind::Match(_, ref arms, _) => {
                let arms_n: u64 = arms.iter().map(|arm| arm.pats.len() as u64).sum();
                if arms_n > 1 {
                    self.cc += 1;
                }
                self.cc += arms.iter().filter(|arm| arm.guard.is_some()).count() as u64;
            },
            ExprKind::Ret(_) => self.returns += 1,
            _ => {},
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
