// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module implements the check that the lifetime of a borrow
//! does not exceed the lifetime of the value being borrowed.

use borrowck::*;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::region;
use rustc::middle::ty;
use rustc::util::ppaux::Repr;
use syntax::ast;
use syntax::codemap::Span;

type R = Result<(),()>;

pub fn guarantee_lifetime<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                    item_scope: region::CodeExtent,
                                    span: Span,
                                    cause: euv::LoanCause,
                                    cmt: mc::cmt<'tcx>,
                                    loan_region: ty::Region,
                                    _: ty::BorrowKind)
                                    -> Result<(),()> {
    //! Reports error if `loan_region` is larger than S
    //! where S is `item_scope` if `cmt` is an upvar,
    //! and is scope of `cmt` otherwise.
    debug!("guarantee_lifetime(cmt={}, loan_region={})",
           cmt.repr(bccx.tcx), loan_region.repr(bccx.tcx));
    let ctxt = GuaranteeLifetimeContext {bccx: bccx,
                                         item_scope: item_scope,
                                         span: span,
                                         cause: cause,
                                         loan_region: loan_region,
                                         cmt_original: cmt.clone()};
    ctxt.check(&cmt, None)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct GuaranteeLifetimeContext<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,

    // the scope of the function body for the enclosing item
    item_scope: region::CodeExtent,

    span: Span,
    cause: euv::LoanCause,
    loan_region: ty::Region,
    cmt_original: mc::cmt<'tcx>
}

impl<'a, 'tcx> GuaranteeLifetimeContext<'a, 'tcx> {

    fn check(&self, cmt: &mc::cmt<'tcx>, discr_scope: Option<ast::NodeId>) -> R {
        //! Main routine. Walks down `cmt` until we find the
        //! "guarantor".  Reports an error if `self.loan_region` is
        //! larger than scope of `cmt`.
        debug!("guarantee_lifetime.check(cmt={}, loan_region={})",
               cmt.repr(self.bccx.tcx),
               self.loan_region.repr(self.bccx.tcx));

        match cmt.cat {
            mc::cat_rvalue(..) |
            mc::cat_local(..) |                         // L-Local
            mc::cat_upvar(..) |
            mc::cat_deref(_, _, mc::BorrowedPtr(..)) |  // L-Deref-Borrowed
            mc::cat_deref(_, _, mc::Implicit(..)) |
            mc::cat_deref(_, _, mc::UnsafePtr(..)) => {
                self.check_scope(self.scope(cmt))
            }

            mc::cat_static_item => {
                Ok(())
            }

            mc::cat_downcast(ref base, _) |
            mc::cat_deref(ref base, _, mc::OwnedPtr) |     // L-Deref-Send
            mc::cat_interior(ref base, _) => {             // L-Field
                self.check(base, discr_scope)
            }
        }
    }

    fn check_scope(&self, max_scope: ty::Region) -> R {
        //! Reports an error if `loan_region` is larger than `max_scope`

        if !self.bccx.is_subregion_of(self.loan_region, max_scope) {
            Err(self.report_error(err_out_of_scope(max_scope, self.loan_region)))
        } else {
            Ok(())
        }
    }

    fn scope(&self, cmt: &mc::cmt) -> ty::Region {
        //! Returns the maximal region scope for the which the
        //! lvalue `cmt` is guaranteed to be valid without any
        //! rooting etc, and presuming `cmt` is not mutated.

        // See the SCOPE(LV) function in doc.rs

        match cmt.cat {
            mc::cat_rvalue(temp_scope) => {
                temp_scope
            }
            mc::cat_upvar(..) => {
                ty::ReScope(self.item_scope)
            }
            mc::cat_static_item => {
                ty::ReStatic
            }
            mc::cat_local(local_id) => {
                ty::ReScope(self.bccx.tcx.region_maps.var_scope(local_id))
            }
            mc::cat_deref(_, _, mc::UnsafePtr(..)) => {
                ty::ReStatic
            }
            mc::cat_deref(_, _, mc::BorrowedPtr(_, r)) |
            mc::cat_deref(_, _, mc::Implicit(_, r)) => {
                r
            }
            mc::cat_downcast(ref cmt, _) |
            mc::cat_deref(ref cmt, _, mc::OwnedPtr) |
            mc::cat_interior(ref cmt, _) => {
                self.scope(cmt)
            }
        }
    }

    fn report_error(&self, code: bckerr_code) {
        self.bccx.report(BckError { cmt: self.cmt_original.clone(),
                                    span: self.span,
                                    cause: self.cause,
                                    code: code });
    }
}
