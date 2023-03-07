//! calculate cognitive complexity and warn about overly complex functions

use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{get_async_fn_body, is_async_fn, LimitStack};
use core::ops::ControlFlow;
use rustc_ast::ast::Attribute;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, Expr, ExprKind, FnDecl};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::source_map::Span;
use rustc_span::{sym, BytePos};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for methods with high cognitive complexity.
    ///
    /// ### Why is this bad?
    /// Methods of high cognitive complexity tend to be hard to
    /// both read and maintain. Also LLVM will tend to optimize small methods better.
    ///
    /// ### Known problems
    /// Sometimes it's hard to find a way to reduce the
    /// complexity.
    ///
    /// ### Example
    /// You'll see it when you get the warning.
    #[clippy::version = "1.35.0"]
    pub COGNITIVE_COMPLEXITY,
    nursery,
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
    #[expect(clippy::cast_possible_truncation)]
    fn check<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        expr: &'tcx Expr<'_>,
        body_span: Span,
    ) {
        if body_span.from_expansion() {
            return;
        }

        let mut cc = 1u64;
        let mut returns = 0u64;
        let _: Option<!> = for_each_expr(expr, |e| {
            match e.kind {
                ExprKind::If(_, _, _) => {
                    cc += 1;
                },
                ExprKind::Match(_, arms, _) => {
                    if arms.len() > 1 {
                        cc += 1;
                    }
                    cc += arms.iter().filter(|arm| arm.guard.is_some()).count() as u64;
                },
                ExprKind::Ret(_) => returns += 1,
                _ => {},
            }
            ControlFlow::Continue(())
        });

        let ret_ty = cx.typeck_results().node_type(expr.hir_id);
        let ret_adjust = if is_type_diagnostic_item(cx, ret_ty, sym::Result) {
            returns
        } else {
            #[expect(clippy::integer_division)]
            (returns / 2)
        };

        // prevent degenerate cases where unreachable code contains `return` statements
        if cc >= ret_adjust {
            cc -= ret_adjust;
        }

        if cc > self.limit.limit() {
            let fn_span = match kind {
                FnKind::ItemFn(ident, _, _) | FnKind::Method(ident, _) => ident.span,
                FnKind::Closure => {
                    let header_span = body_span.with_hi(decl.output.span().lo());
                    let pos = snippet_opt(cx, header_span).and_then(|snip| {
                        let low_offset = snip.find('|')?;
                        let high_offset = 1 + snip.get(low_offset + 1..)?.find('|')?;
                        let low = header_span.lo() + BytePos(low_offset as u32);
                        let high = low + BytePos(high_offset as u32 + 1);

                        Some((low, high))
                    });

                    if let Some((low, high)) = pos {
                        Span::new(low, high, header_span.ctxt(), header_span.parent())
                    } else {
                        return;
                    }
                },
            };

            span_lint_and_help(
                cx,
                COGNITIVE_COMPLEXITY,
                fn_span,
                &format!(
                    "the function has a cognitive complexity of ({cc}/{})",
                    self.limit.limit()
                ),
                None,
                "you could split it up into multiple smaller functions",
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for CognitiveComplexity {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        if !cx.tcx.has_attr(def_id.to_def_id(), sym::test) {
            let expr = if is_async_fn(kind) {
                match get_async_fn_body(cx.tcx, body) {
                    Some(b) => b,
                    None => {
                        return;
                    },
                }
            } else {
                body.value
            };

            self.check(cx, kind, decl, expr, span);
        }
    }

    fn enter_lint_attrs(&mut self, cx: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.push_attrs(cx.sess(), attrs, "cognitive_complexity");
    }
    fn exit_lint_attrs(&mut self, cx: &LateContext<'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.pop_attrs(cx.sess(), attrs, "cognitive_complexity");
    }
}
