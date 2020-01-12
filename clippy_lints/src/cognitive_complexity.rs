//! calculate cognitive complexity and warn about overly complex functions

use rustc::hir::map::Map;
use rustc_hir::intravisit::{walk_expr, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
use rustc_span::BytePos;
use syntax::ast::Attribute;

use crate::utils::{match_type, paths, snippet_opt, span_help_and_lint, LimitStack};

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
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            limit: LimitStack::new(limit),
        }
    }
}

impl_lint_pass!(CognitiveComplexity => [COGNITIVE_COMPLEXITY]);

impl CognitiveComplexity {
    #[allow(clippy::cast_possible_truncation)]
    fn check<'a, 'tcx>(
        &mut self,
        cx: &'a LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        body_span: Span,
    ) {
        if body_span.from_expansion() {
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
            let fn_span = match kind {
                FnKind::ItemFn(ident, _, _, _, _) | FnKind::Method(ident, _, _, _) => ident.span,
                FnKind::Closure(_) => {
                    let header_span = body_span.with_hi(decl.output.span().lo());
                    let pos = snippet_opt(cx, header_span).and_then(|snip| {
                        let low_offset = snip.find('|')?;
                        let high_offset = 1 + snip.get(low_offset + 1..)?.find('|')?;
                        let low = header_span.lo() + BytePos(low_offset as u32);
                        let high = low + BytePos(high_offset as u32 + 1);

                        Some((low, high))
                    });

                    if let Some((low, high)) = pos {
                        Span::new(low, high, header_span.ctxt())
                    } else {
                        return;
                    }
                },
            };

            span_help_and_lint(
                cx,
                COGNITIVE_COMPLEXITY,
                fn_span,
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
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        hir_id: HirId,
    ) {
        let def_id = cx.tcx.hir().local_def_id(hir_id);
        if !cx.tcx.has_attr(def_id, sym!(test)) {
            self.check(cx, kind, decl, body, span);
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
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        walk_expr(self, e);
        match e.kind {
            ExprKind::Match(_, ref arms, _) => {
                if arms.len() > 1 {
                    self.cc += 1;
                }
                self.cc += arms.iter().filter(|arm| arm.guard.is_some()).count() as u64;
            },
            ExprKind::Ret(_) => self.returns += 1,
            _ => {},
        }
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::None
    }
}
