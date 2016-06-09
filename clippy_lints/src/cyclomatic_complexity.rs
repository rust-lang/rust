//! calculate cyclomatic complexity and warn about overly complex functions

use rustc::cfg::CFG;
use rustc::lint::*;
use rustc::ty;
use rustc::hir::*;
use rustc::hir::intravisit::{Visitor, walk_expr};
use syntax::ast::Attribute;
use syntax::attr;
use syntax::codemap::Span;

use utils::{in_macro, LimitStack, span_help_and_lint, paths, match_type};

/// **What it does:** This lint checks for methods with high cyclomatic complexity
///
/// **Why is this bad?** Methods of high cyclomatic complexity tend to be badly readable. Also LLVM
/// will usually optimize small methods better.
///
/// **Known problems:** Sometimes it's hard to find a way to reduce the complexity
///
/// **Example:** No. You'll see it when you get the warning.
declare_lint! {
    pub CYCLOMATIC_COMPLEXITY, Warn,
    "finds functions that should be split up into multiple functions"
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
    fn check<'a, 'tcx>(&mut self, cx: &'a LateContext<'a, 'tcx>, block: &Block, span: Span) {
        if in_macro(cx, span) {
            return;
        }

        let cfg = CFG::new(cx.tcx, block);
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
            tcx: &cx.tcx,
        };
        helper.visit_block(block);
        let CCHelper { match_arms, divergence, short_circuits, returns, .. } = helper;
        let ret_ty = cx.tcx.node_id_to_type(block.id);
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
                span_help_and_lint(cx,
                                   CYCLOMATIC_COMPLEXITY,
                                   span,
                                   &format!("the function has a cyclomatic complexity of {}", rust_cc),
                                   "you could split it up into multiple smaller functions");
            }
        }
    }
}

impl LateLintPass for CyclomaticComplexity {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemFn(_, _, _, _, _, ref block) = item.node {
            if !attr::contains_name(&item.attrs, "test") {
                self.check(cx, block, item.span);
            }
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

struct CCHelper<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    match_arms: u64,
    divergence: u64,
    returns: u64,
    short_circuits: u64, // && and ||
    tcx: &'a ty::TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'b, 'tcx, 'gcx> Visitor<'a> for CCHelper<'b, 'gcx, 'tcx> {
    fn visit_expr(&mut self, e: &'a Expr) {
        match e.node {
            ExprMatch(_, ref arms, _) => {
                walk_expr(self, e);
                let arms_n: u64 = arms.iter().map(|arm| arm.pats.len() as u64).sum();
                if arms_n > 1 {
                    self.match_arms += arms_n - 2;
                }
            }
            ExprCall(ref callee, _) => {
                walk_expr(self, e);
                let ty = self.tcx.node_id_to_type(callee.id);
                match ty.sty {
                    ty::TyFnDef(_, _, ty) |
                    ty::TyFnPtr(ty) if ty.sig.skip_binder().output.diverges() => {
                        self.divergence += 1;
                    }
                    _ => (),
                }
            }
            ExprClosure(..) => (),
            ExprBinary(op, _, _) => {
                walk_expr(self, e);
                match op.node {
                    BiAnd | BiOr => self.short_circuits += 1,
                    _ => (),
                }
            }
            ExprRet(_) => self.returns += 1,
            _ => walk_expr(self, e),
        }
    }
}

#[cfg(feature="debugging")]
fn report_cc_bug(_: &LateContext, cc: u64, narms: u64, div: u64, shorts: u64, returns: u64, span: Span) {
    span_bug!(span,
              "Clippy encountered a bug calculating cyclomatic complexity: cc = {}, arms = {}, \
               div = {}, shorts = {}, returns = {}. Please file a bug report.",
              cc,
              narms,
              div,
              shorts,
              returns);
}
#[cfg(not(feature="debugging"))]
fn report_cc_bug(cx: &LateContext, cc: u64, narms: u64, div: u64, shorts: u64, returns: u64, span: Span) {
    if cx.current_level(CYCLOMATIC_COMPLEXITY) != Level::Allow {
        cx.sess().span_note_without_error(span,
                                          &format!("Clippy encountered a bug calculating cyclomatic complexity \
                                                    (hide this message with `#[allow(cyclomatic_complexity)]`): \
                                                    cc = {}, arms = {}, div = {}, shorts = {}, returns = {}. \
                                                    Please file a bug report.",
                                                   cc,
                                                   narms,
                                                   div,
                                                   shorts,
                                                   returns));
    }
}
