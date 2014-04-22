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

use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty;
use util::ppaux::ty_to_string;

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit;

pub fn check_crate(tcx: &ty::ctxt,
                   krate: &ast::Crate) {
    let mut rvcx = RvalueContext { tcx: tcx };
    visit::walk_crate(&mut rvcx, krate, ());
}

struct RvalueContext<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>
}

impl<'a, 'tcx> visit::Visitor<()> for RvalueContext<'a, 'tcx> {
    fn visit_fn(&mut self,
                _: &visit::FnKind,
                fd: &ast::FnDecl,
                b: &ast::Block,
                _: Span,
                _: ast::NodeId,
                _: ()) {
        let mut euv = euv::ExprUseVisitor::new(self, self.tcx);
        euv.walk_fn(fd, b);
    }
}

impl<'a, 'tcx> euv::Delegate for RvalueContext<'a, 'tcx> {
    fn consume(&mut self,
               _: ast::NodeId,
               span: Span,
               cmt: mc::cmt,
               _: euv::ConsumeMode) {
        debug!("consume; cmt: {:?}; type: {}", *cmt, ty_to_string(self.tcx, cmt.ty));
        if !ty::type_is_sized(self.tcx, cmt.ty) {
            span_err!(self.tcx.sess, span, E0161,
                "cannot move a value of type {0}: the size of {0} cannot be statically determined",
                ty_to_string(self.tcx, cmt.ty));
        }
    }

    fn consume_pat(&mut self,
                   _consume_pat: &ast::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {
    }

    fn borrow(&mut self,
              _borrow_id: ast::NodeId,
              _borrow_span: Span,
              _cmt: mc::cmt,
              _loan_region: ty::Region,
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
