// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::intravisit as visit;
use crate::rustc::hir::*;
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::middle::expr_use_visitor::*;
use crate::rustc::middle::mem_categorization::{cmt_, Categorization};
use crate::rustc::ty::layout::LayoutOf;
use crate::rustc::ty::{self, Ty};
use crate::rustc::util::nodemap::NodeSet;
use crate::rustc::{declare_tool_lint, lint_array};
use crate::syntax::ast::NodeId;
use crate::syntax::source_map::Span;
use crate::utils::span_lint;

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
declare_clippy_lint! {
    pub BOXED_LOCAL,
    perf,
    "using `Box<T>` where unnecessary"
}

fn is_non_trait_box(ty: Ty<'_>) -> bool {
    ty.is_box() && !ty.boxed_ty().is_trait()
}

struct EscapeDelegate<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    set: NodeSet,
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
        node_id: NodeId,
    ) {
        // If the method is an impl for a trait, don't warn
        let parent_id = cx.tcx.hir.get_parent(node_id);
        let parent_node = cx.tcx.hir.find(parent_id);

        if let Some(Node::Item(item)) = parent_node {
            if let ItemKind::Impl(_, _, _, _, Some(..), _, _) = item.node {
                return;
            }
        }

        let mut v = EscapeDelegate {
            cx,
            set: NodeSet::default(),
            too_large_for_stack: self.too_large_for_stack,
        };

        let fn_def_id = cx.tcx.hir.local_def_id(node_id);
        let region_scope_tree = &cx.tcx.region_scope_tree(fn_def_id);
        ExprUseVisitor::new(&mut v, cx.tcx, cx.param_env, region_scope_tree, cx.tables, None).consume_body(body);

        for node in v.set {
            span_lint(
                cx,
                BOXED_LOCAL,
                cx.tcx.hir.span(node),
                "local variable doesn't need to be boxed here",
            );
        }
    }
}

impl<'a, 'tcx> Delegate<'tcx> for EscapeDelegate<'a, 'tcx> {
    fn consume(&mut self, _: NodeId, _: Span, cmt: &cmt_<'tcx>, mode: ConsumeMode) {
        if let Categorization::Local(lid) = cmt.cat {
            if let Move(DirectRefMove) = mode {
                // moved out or in. clearly can't be localized
                self.set.remove(&lid);
            }
        }
    }
    fn matched_pat(&mut self, _: &Pat, _: &cmt_<'tcx>, _: MatchMode) {}
    fn consume_pat(&mut self, consume_pat: &Pat, cmt: &cmt_<'tcx>, _: ConsumeMode) {
        let map = &self.cx.tcx.hir;
        if map.is_argument(consume_pat.id) {
            // Skip closure arguments
            if let Some(Node::Expr(..)) = map.find(map.get_parent_node(consume_pat.id)) {
                return;
            }
            if is_non_trait_box(cmt.ty) && !self.is_large_box(cmt.ty) {
                self.set.insert(consume_pat.id);
            }
            return;
        }
        if let Categorization::Rvalue(..) = cmt.cat {
            let id = map.hir_to_node_id(cmt.hir_id);
            if let Some(Node::Stmt(st)) = map.find(map.get_parent_node(id)) {
                if let StmtKind::Decl(ref decl, _) = st.node {
                    if let DeclKind::Local(ref loc) = decl.node {
                        if let Some(ref ex) = loc.init {
                            if let ExprKind::Box(..) = ex.node {
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
        cmt: &cmt_<'tcx>,
        _: ty::Region<'_>,
        _: ty::BorrowKind,
        loan_cause: LoanCause,
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
    fn mutate(&mut self, _: NodeId, _: Span, _: &cmt_<'tcx>, _: MutateMode) {}
}

impl<'a, 'tcx> EscapeDelegate<'a, 'tcx> {
    fn is_large_box(&self, ty: Ty<'tcx>) -> bool {
        // Large types need to be boxed to avoid stack
        // overflows.
        if ty.is_box() {
            self.cx.layout_of(ty.boxed_ty()).ok().map_or(0, |l| l.size.bytes()) > self.too_large_for_stack
        } else {
            false
        }
    }
}
