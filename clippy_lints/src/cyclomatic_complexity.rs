//! calculate cyclomatic complexity and warn about overly complex functions

use rustc::cfg::CFG;
use rustc::lint::*;
use rustc::hir::*;
use rustc::ty;
use rustc::hir::intravisit::{Visitor, walk_expr, NestedVisitorMap};
use syntax::ast::{Attribute, NodeId};
use syntax::codemap::Span;

use utils::{in_macro, LimitStack, span_help_and_lint, paths, match_type};

/// **What it does:** Checks for methods with high cyclomatic complexity.
///
/// **Why is this bad?** Methods of high cyclomatic complexity tend to be badly
/// readable. Also LLVM will usually optimize small methods better.
///
/// **Known problems:** Sometimes it's hard to find a way to reduce the
/// complexity.
///
/// **Example:** No. You'll see it when you get the warning.
declare_lint! {
    pub CYCLOMATIC_COMPLEXITY,
    Warn,
    "functions that should be split up into multiple functions"
}

pub struct CyclomaticComplexity {
    limit: LimitStack,
}

impl CyclomaticComplexity {
    pub fn new(limit: u64) -> Self {
        CyclomaticComplexity { limit: LimitStack::new(limit) }
    }
}

impl LintPass for CyclomaticComplexity {
    fn get_lints(&self) -> LintArray {
        lint_array!(CYCLOMATIC_COMPLEXITY)
    }
}

impl CyclomaticComplexity {
    fn check<'a, 'tcx: 'a>(&mut self, cx: &'a LateContext<'a, 'tcx>, body: &'tcx Body, span: Span) {
        if in_macro(span) {
            return;
        }

        let cfg = CFG::new(cx.tcx, body);
        let expr = &body.value;
        let n = cfg.graph.len_nodes() as u64;
        let e = cfg.graph.len_edges() as u64;
        if e + 2 < n {
            // the function has unreachable code, other lints should catch this
            return;
        }
        let cc = e + 2 - n;
        let mut helper = CCHelper {
            match_arms: 0,
            divergence: 0,
            short_circuits: 0,
            returns: 0,
            cx: cx,
        };
        helper.visit_expr(expr);
        let CCHelper {
            match_arms,
            divergence,
            short_circuits,
            returns,
            ..
        } = helper;
        let ret_ty = cx.tables.node_id_to_type(expr.id);
        let ret_adjust = if match_type(cx, ret_ty, &paths::RESULT) {
            returns
        } else {
            returns / 2
        };

        if cc + divergence < match_arms + short_circuits {
            report_cc_bug(cx, cc, match_arms, divergence, short_circuits, ret_adjust, span);
        } else {
            let mut rust_cc = cc + divergence - match_arms - short_circuits;
            // prevent degenerate cases where unreachable code contains `return` statements
            if rust_cc >= ret_adjust {
                rust_cc -= ret_adjust;
            }
            if rust_cc > self.limit.limit() {
                span_help_and_lint(
                    cx,
                    CYCLOMATIC_COMPLEXITY,
                    span,
                    &format!("the function has a cyclomatic complexity of {}", rust_cc),
                    "you could split it up into multiple smaller functions",
                );
            }
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CyclomaticComplexity {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        node_id: NodeId,
    ) {
        let def_id = cx.tcx.hir.local_def_id(node_id);
        if !cx.tcx.has_attr(def_id, "test") {
            self.check(cx, body, span);
        }
    }

    fn enter_lint_attrs(&mut self, cx: &LateContext<'a, 'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.push_attrs(
            cx.sess(),
            attrs,
            "cyclomatic_complexity",
        );
    }
    fn exit_lint_attrs(&mut self, cx: &LateContext<'a, 'tcx>, attrs: &'tcx [Attribute]) {
        self.limit.pop_attrs(
            cx.sess(),
            attrs,
            "cyclomatic_complexity",
        );
    }
}

struct CCHelper<'a, 'tcx: 'a> {
    match_arms: u64,
    divergence: u64,
    returns: u64,
    short_circuits: u64, // && and ||
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for CCHelper<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr) {
        match e.node {
            ExprMatch(_, ref arms, _) => {
                walk_expr(self, e);
                let arms_n: u64 = arms.iter().map(|arm| arm.pats.len() as u64).sum();
                if arms_n > 1 {
                    self.match_arms += arms_n - 2;
                }
            },
            ExprCall(ref callee, _) => {
                walk_expr(self, e);
                let ty = self.cx.tables.node_id_to_type(callee.id);
                match ty.sty {
                    ty::TyFnDef(..) | ty::TyFnPtr(_) => {
                        let sig = ty.fn_sig(self.cx.tcx);
                        if sig.skip_binder().output().sty == ty::TyNever {
                            self.divergence += 1;
                        }
                    },
                    _ => (),
                }
            },
            ExprClosure(..) => (),
            ExprBinary(op, _, _) => {
                walk_expr(self, e);
                match op.node {
                    BiAnd | BiOr => self.short_circuits += 1,
                    _ => (),
                }
            },
            ExprRet(_) => self.returns += 1,
            _ => walk_expr(self, e),
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

#[cfg(feature = "debugging")]
fn report_cc_bug(_: &LateContext, cc: u64, narms: u64, div: u64, shorts: u64, returns: u64, span: Span) {
    span_bug!(
        span,
        "Clippy encountered a bug calculating cyclomatic complexity: cc = {}, arms = {}, \
               div = {}, shorts = {}, returns = {}. Please file a bug report.",
        cc,
        narms,
        div,
        shorts,
        returns
    );
}
#[cfg(not(feature = "debugging"))]
fn report_cc_bug(cx: &LateContext, cc: u64, narms: u64, div: u64, shorts: u64, returns: u64, span: Span) {
    if cx.current_level(CYCLOMATIC_COMPLEXITY) != Level::Allow {
        cx.sess().span_note_without_error(
            span,
            &format!(
                "Clippy encountered a bug calculating cyclomatic complexity \
                                                    (hide this message with `#[allow(cyclomatic_complexity)]`): \
                                                    cc = {}, arms = {}, div = {}, shorts = {}, returns = {}. \
                                                    Please file a bug report.",
                cc,
                narms,
                div,
                shorts,
                returns
            ),
        );
    }
}
