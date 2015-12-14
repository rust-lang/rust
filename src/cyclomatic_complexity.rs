//! calculate cyclomatic complexity and warn about overly complex functions

use rustc::lint::*;
use rustc_front::hir::*;
use rustc::middle::cfg::CFG;
use rustc::middle::ty;
use syntax::codemap::Span;
use syntax::attr::*;
use syntax::ast::Attribute;
use rustc_front::intravisit::{Visitor, walk_expr};

use utils::{in_macro, LimitStack};

declare_lint! { pub CYCLOMATIC_COMPLEXITY, Warn,
    "finds functions that should be split up into multiple functions" }

pub struct CyclomaticComplexity {
    limit: LimitStack,
}

impl CyclomaticComplexity {
    pub fn new(limit: u64) -> Self {
        CyclomaticComplexity {
            limit: LimitStack::new(limit),
        }
    }
}

impl LintPass for CyclomaticComplexity {
    fn get_lints(&self) -> LintArray {
        lint_array!(CYCLOMATIC_COMPLEXITY)
    }
}

impl CyclomaticComplexity {
    fn check<'a, 'tcx>(&mut self, cx: &'a LateContext<'a, 'tcx>, block: &Block, span: Span) {
        if in_macro(cx, span) { return; }
        let cfg = CFG::new(cx.tcx, block);
        let n = cfg.graph.len_nodes() as u64;
        let e = cfg.graph.len_edges() as u64;
        let cc = e + 2 - n;
        let mut arm_counter = MatchArmCounter(0);
        arm_counter.visit_block(block);
        let narms = arm_counter.0;

        let mut diverge_counter = DivergenceCounter(0, &cx.tcx);
        diverge_counter.visit_block(block);
        let divergence = diverge_counter.0;

        if cc + divergence < narms {
            report_cc_bug(cx, cc, narms, divergence, span);
        } else {
            let rust_cc = cc + divergence - narms;
            if rust_cc > self.limit.limit() {
                cx.span_lint_help(CYCLOMATIC_COMPLEXITY, span,
                &format!("The function has a cyclomatic complexity of {}.", rust_cc),
                "You could split it up into multiple smaller functions");
            }
        }
    }
}

impl LateLintPass for CyclomaticComplexity {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemFn(_, _, _, _, _, ref block) = item.node {
            self.check(cx, block, item.span);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext, item: &ImplItem) {
        if let ImplItemKind::Method(_, ref block) = item.node {
            self.check(cx, block, item.span);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &TraitItem) {
        if let MethodTraitItem(_, Some(ref block)) = item.node {
            self.check(cx, block, item.span);
        }
    }

    fn enter_lint_attrs(&mut self, cx: &LateContext, attrs: &[Attribute]) {
        self.limit.push_attrs(cx.sess(), attrs, "cyclomatic_complexity");
    }
    fn exit_lint_attrs(&mut self, cx: &LateContext, attrs: &[Attribute]) {
        self.limit.pop_attrs(cx.sess(), attrs, "cyclomatic_complexity");
    }
}

struct MatchArmCounter(u64);

impl<'a> Visitor<'a> for MatchArmCounter {
    fn visit_expr(&mut self, e: &'a Expr) {
        match e.node {
            ExprMatch(_, ref arms, _) => {
                walk_expr(self, e);
                let arms_n: u64 = arms.iter().map(|arm| arm.pats.len() as u64).sum();
                if arms_n > 1 {
                    self.0 += arms_n - 2;
                }
            },
            ExprClosure(..) => {},
            _ => walk_expr(self, e),
        }
    }
}

struct DivergenceCounter<'a, 'tcx: 'a>(u64, &'a ty::ctxt<'tcx>);

impl<'a, 'b, 'tcx> Visitor<'a> for DivergenceCounter<'b, 'tcx> {
    fn visit_expr(&mut self, e: &'a Expr) {
        match e.node {
            ExprCall(ref callee, _) => {
                walk_expr(self, e);
                let ty = self.1.node_id_to_type(callee.id);
                if let ty::TyBareFn(_, ty) = ty.sty {
                    if ty.sig.skip_binder().output.diverges() {
                        self.0 += 1;
                    }
                }
            },
            ExprClosure(..) => {},
            _ => walk_expr(self, e),
        }
    }
}

#[cfg(feature="debugging")]
fn report_cc_bug(cx: &LateContext, cc: u64, narms: u64, div: u64, span: Span) {
    cx.sess().span_bug(span, &format!("Clippy encountered a bug calculating cyclomatic complexity: \
                                       cc = {}, arms = {}, div = {}. Please file a bug report.", cc, narms, div));;
}
#[cfg(not(feature="debugging"))]
fn report_cc_bug(cx: &LateContext, cc: u64, narms: u64, div: u64, span: Span) {
    if cx.current_level(CYCLOMATIC_COMPLEXITY) != Level::Allow {
        cx.sess().span_note(span, &format!("Clippy encountered a bug calculating cyclomatic complexity \
                                            (hide this message with `#[allow(cyclomatic_complexity)]`): \
                                            cc = {}, arms = {}, div = {}. Please file a bug report.", cc, narms, div));
    }
}
