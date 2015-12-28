use rustc::lint::*;
use rustc_front::hir::*;
use rustc_front::intravisit as visit;
use rustc::front::map::Node;
use rustc::middle::ty;
use rustc::middle::ty::adjustment::AutoAdjustment;
use rustc::middle::expr_use_visitor::*;
use rustc::middle::infer;
use rustc::middle::mem_categorization::{cmt, Categorization};
use rustc::util::nodemap::NodeSet;
use syntax::ast::NodeId;
use syntax::codemap::Span;
use utils::span_lint;

pub struct EscapePass;

/// **What it does:** This lint checks for usage of `Box<T>` where an unboxed `T` would work fine
///
/// **Why is this bad?** This is an unnecessary allocation, and bad for performance
///
/// It is only necessary to allocate if you wish to move the box into something.
///
/// **Example:**
///
/// ```rust
/// fn main() {
///     let x = Box::new(1);
///     foo(*x);
///     println!("{}", *x);
/// }
/// ```
declare_lint!(pub BOXED_LOCAL, Warn, "using Box<T> where unnecessary");

fn is_box(ty: ty::Ty) -> bool {
    match ty.sty {
        ty::TyBox(..) => true,
        _ => false
    }
}

struct EscapeDelegate<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    set: NodeSet,
}

impl LintPass for EscapePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOXED_LOCAL)
    }
}

impl LateLintPass for EscapePass {
    fn check_fn(&mut self,
                cx: &LateContext,
                _: visit::FnKind,
                decl: &FnDecl,
                body: &Block,
                _: Span,
                id: NodeId) {
        let param_env = ty::ParameterEnvironment::for_item(cx.tcx, id);
        let infcx = infer::new_infer_ctxt(cx.tcx, &cx.tcx.tables, Some(param_env), false);
        let mut v = EscapeDelegate {
            cx: cx,
            set: NodeSet(),
        };
        {
            let mut vis = ExprUseVisitor::new(&mut v, &infcx);
            vis.walk_fn(decl, body);
        }
        for node in v.set {
            span_lint(cx,
                      BOXED_LOCAL,
                      cx.tcx.map.span(node),
                      "local variable doesn't need to be boxed here");
        }
    }
}

impl<'a, 'tcx: 'a> Delegate<'tcx> for EscapeDelegate<'a, 'tcx> {
    fn consume(&mut self,
               _: NodeId,
               _: Span,
               cmt: cmt<'tcx>,
               mode: ConsumeMode) {

        if let Categorization::Local(lid) = cmt.cat {
            if self.set.contains(&lid) {
                if let Move(DirectRefMove) = mode {
                    // moved out or in. clearly can't be localized
                    self.set.remove(&lid);
                }
            }
        }
    }
    fn matched_pat(&mut self, _: &Pat, _: cmt<'tcx>, _: MatchMode) {}
    fn consume_pat(&mut self, consume_pat: &Pat, cmt: cmt<'tcx>, _: ConsumeMode) {
        if self.cx.tcx.map.is_argument(consume_pat.id) {
            if is_box(cmt.ty) {
                self.set.insert(consume_pat.id);
            }
            return;
        }
        if let Categorization::Rvalue(..) = cmt.cat {
            if let Some(Node::NodeStmt(st)) = self.cx
                                                  .tcx
                                                  .map
                                                  .find(self.cx.tcx.map.get_parent_node(cmt.id)) {
                if let StmtDecl(ref decl, _) = st.node {
                    if let DeclLocal(ref loc) = decl.node {
                        if let Some(ref ex) = loc.init {
                            if let ExprBox(..) = ex.node {
                                if is_box(cmt.ty) {
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
    fn borrow(&mut self,
              borrow_id: NodeId,
              _: Span,
              cmt: cmt<'tcx>,
              _: ty::Region,
              _: ty::BorrowKind,
              loan_cause: LoanCause) {

        if let Categorization::Local(lid) = cmt.cat {
            if self.set.contains(&lid) {
                if let Some(&AutoAdjustment::AdjustDerefRef(adj)) = self.cx
                                                                        .tcx
                                                                        .tables
                                                                        .borrow()
                                                                        .adjustments
                                                                        .get(&borrow_id) {
                    if LoanCause::AutoRef == loan_cause {
                        // x.foo()
                        if adj.autoderefs <= 0 {
                            self.set.remove(&lid); // Used without autodereffing (i.e. x.clone())
                        }
                    } else {
                        self.cx.sess().span_bug(cmt.span, "Unknown adjusted AutoRef");
                    }
                } else if LoanCause::AddrOf == loan_cause {
                    // &x
                    if let Some(&AutoAdjustment::AdjustDerefRef(adj)) =
                           self.cx.tcx.tables.borrow().adjustments
                               .get(&self.cx.tcx.map.get_parent_node(borrow_id)) {
                        if adj.autoderefs <= 1 {
                            // foo(&x) where no extra autoreffing is happening
                            self.set.remove(&lid);
                        }
                    }

                } else if LoanCause::MatchDiscriminant == loan_cause {
                    self.set.remove(&lid); // `match x` can move
                }
                // do nothing for matches, etc. These can't escape
            }
        }
    }
    fn decl_without_init(&mut self, _: NodeId, _: Span) {}
    fn mutate(&mut self,
              _: NodeId,
              _: Span,
              _: cmt<'tcx>,
              _: MutateMode) {
    }
}
