use rustc::hir::*;
use rustc::hir::intravisit as visit;
use rustc::hir::map::Node::{NodeExpr, NodeStmt};
use rustc::lint::*;
use rustc::middle::expr_use_visitor::*;
use rustc::middle::mem_categorization::{cmt, Categorization};
use rustc::ty;
use rustc::util::nodemap::NodeSet;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use utils::span_lint;

pub struct Pass {
    pub too_large_for_stack: u64,
}

/// **What it does:** Checks for usage of `Box<T>` where an unboxed `T` would
/// work fine.
///
/// **Why is this bad?** This is an unnecessary allocation, and bad for
/// performance. It is only necessary to allocate if you wish to move the box
/// into something.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn main() {
///     let x = Box::new(1);
///     foo(*x);
///     println!("{}", *x);
/// }
/// ```
declare_lint! {
    pub BOXED_LOCAL,
    Warn,
    "using `Box<T>` where unnecessary"
}

fn is_non_trait_box(ty: ty::Ty) -> bool {
    ty.is_box() && !ty.boxed_ty().is_trait()
}

struct EscapeDelegate<'a, 'tcx: 'a> {
    set: NodeSet,
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    too_large_for_stack: u64,
}

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOXED_LOCAL)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: visit::FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        node_id: NodeId
    ) {
        let fn_def_id = cx.tcx.hir.local_def_id(node_id);
        let param_env = cx.tcx.param_env(fn_def_id).reveal_all();
        let mut v = EscapeDelegate {
            set: NodeSet(),
            tcx: cx.tcx,
            param_env: param_env,
            too_large_for_stack: self.too_large_for_stack,
        };

        cx.tcx.infer_ctxt(body.id()).enter(|infcx| {
            let region_maps = &cx.tcx.region_maps(fn_def_id);
            let mut vis = ExprUseVisitor::new(&mut v, region_maps, &infcx, param_env);
            vis.consume_body(body);
        });

        for node in v.set {
            span_lint(cx,
                      BOXED_LOCAL,
                      cx.tcx.hir.span(node),
                      "local variable doesn't need to be boxed here");
        }
    }
}

impl<'a, 'gcx: 'tcx, 'tcx> Delegate<'tcx> for EscapeDelegate<'a, 'gcx> {
    fn consume(&mut self, _: NodeId, _: Span, cmt: cmt<'tcx>, mode: ConsumeMode) {
        if let Categorization::Local(lid) = cmt.cat {
            if let Move(DirectRefMove) = mode {
                // moved out or in. clearly can't be localized
                self.set.remove(&lid);
            }
        }
    }
    fn matched_pat(&mut self, _: &Pat, _: cmt<'tcx>, _: MatchMode) {}
    fn consume_pat(&mut self, consume_pat: &Pat, cmt: cmt<'tcx>, _: ConsumeMode) {
        let map = &self.tcx.hir;
        if map.is_argument(consume_pat.id) {
            // Skip closure arguments
            if let Some(NodeExpr(..)) = map.find(map.get_parent_node(consume_pat.id)) {
                return;
            }
            if is_non_trait_box(cmt.ty) && !self.is_large_box(cmt.ty) {
                self.set.insert(consume_pat.id);
            }
            return;
        }
        if let Categorization::Rvalue(..) = cmt.cat {
            if let Some(NodeStmt(st)) = map.find(map.get_parent_node(cmt.id)) {
                if let StmtDecl(ref decl, _) = st.node {
                    if let DeclLocal(ref loc) = decl.node {
                        if let Some(ref ex) = loc.init {
                            if let ExprBox(..) = ex.node {
                                if is_non_trait_box(cmt.ty) && !self.is_large_box(cmt.ty) {
                                    // let x = box (...)
                                    self.set.insert(consume_pat.id);
                                }
                                // TODO Box::new
                                // TODO vec![]
                                // TODO "foo".to_owned() and friends
                            }
                        }
                    }
                }
            }
        }
        if let Categorization::Local(lid) = cmt.cat {
            if self.set.contains(&lid) {
                // let y = x where x is known
                // remove x, insert y
                self.set.insert(consume_pat.id);
                self.set.remove(&lid);
            }
        }

    }
    fn borrow(
        &mut self,
        _: NodeId,
        _: Span,
        cmt: cmt<'tcx>,
        _: ty::Region,
        _: ty::BorrowKind,
        loan_cause: LoanCause
    ) {
        if let Categorization::Local(lid) = cmt.cat {
            match loan_cause {
                // x.foo()
                // Used without autodereffing (i.e. x.clone())
                LoanCause::AutoRef |

                // &x
                // foo(&x) where no extra autoreffing is happening
                LoanCause::AddrOf |

                // `match x` can move
                LoanCause::MatchDiscriminant => {
                    self.set.remove(&lid);
                }

                // do nothing for matches, etc. These can't escape
                _ => {}
            }
        }
    }
    fn decl_without_init(&mut self, _: NodeId, _: Span) {}
    fn mutate(&mut self, _: NodeId, _: Span, _: cmt<'tcx>, _: MutateMode) {}
}

impl<'a, 'tcx: 'a> EscapeDelegate<'a, 'tcx> {
    fn is_large_box(&self, ty: ty::Ty) -> bool {
        // Large types need to be boxed to avoid stack
        // overflows.
        if ty.is_box() {
            if let Some(inner) = self.tcx.lift(&ty.boxed_ty()) {
                if let Ok(layout) = inner.layout(self.tcx, self.param_env) {
                    return layout.size(self.tcx).bytes() > self.too_large_for_stack;
                }
            }
        }

        false
    }
}
