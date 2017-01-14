// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that all rvalues in a crate have statically known size. check_crate
// is the public starting point.

use rustc::dep_graph::DepNode;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::ty::{self, TyCtxt};
use rustc::traits::Reveal;

use rustc::hir;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use syntax::ast;
use syntax_pos::Span;

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut rvcx = RvalueContext { tcx: tcx };
    tcx.visit_all_item_likes_in_krate(DepNode::RvalueCheck, &mut rvcx.as_deep_visitor());
}

struct RvalueContext<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for RvalueContext<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let body = self.tcx.map.body(body_id);
        self.tcx.infer_ctxt(body_id, Reveal::NotSpecializable).enter(|infcx| {
            let mut delegate = RvalueContextDelegate {
                tcx: infcx.tcx,
                param_env: &infcx.parameter_environment
            };
            euv::ExprUseVisitor::new(&mut delegate, &infcx).consume_body(body);
        });
        self.visit_body(body);
    }
}

struct RvalueContextDelegate<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    param_env: &'a ty::ParameterEnvironment<'gcx>,
}

impl<'a, 'gcx, 'tcx> euv::Delegate<'tcx> for RvalueContextDelegate<'a, 'gcx, 'tcx> {
    fn consume(&mut self,
               _: ast::NodeId,
               span: Span,
               cmt: mc::cmt<'tcx>,
               _: euv::ConsumeMode) {
        debug!("consume; cmt: {:?}; type: {:?}", *cmt, cmt.ty);
        let ty = self.tcx.lift_to_global(&cmt.ty).unwrap();
        if !ty.is_sized(self.tcx.global_tcx(), self.param_env, span) {
            span_err!(self.tcx.sess, span, E0161,
                "cannot move a value of type {0}: the size of {0} cannot be statically determined",
                ty);
        }
    }

    fn matched_pat(&mut self,
                   _matched_pat: &hir::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::MatchMode) {}

    fn consume_pat(&mut self,
                   _consume_pat: &hir::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {
    }

    fn borrow(&mut self,
              _borrow_id: ast::NodeId,
              _borrow_span: Span,
              _cmt: mc::cmt,
              _loan_region: &'tcx ty::Region,
              _bk: ty::BorrowKind,
              _loan_cause: euv::LoanCause) {
    }

    fn decl_without_init(&mut self,
                         _id: ast::NodeId,
                         _span: Span) {
    }

    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              _assignee_cmt: mc::cmt,
              _mode: euv::MutateMode) {
    }
}
