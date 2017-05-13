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
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::ty;

use syntax::ast;
use syntax_pos::Span;

type R = Result<(),()>;

pub fn guarantee_lifetime<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                    item_scope: region::CodeExtent,
                                    span: Span,
                                    cause: euv::LoanCause,
                                    cmt: mc::cmt<'tcx>,
                                    loan_region: &'tcx ty::Region,
                                    _: ty::BorrowKind)
                                    -> Result<(),()> {
    //! Reports error if `loan_region` is larger than S
    //! where S is `item_scope` if `cmt` is an upvar,
    //! and is scope of `cmt` otherwise.
    debug!("guarantee_lifetime(cmt={:?}, loan_region={:?})",
           cmt, loan_region);
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
    loan_region: &'tcx ty::Region,
    cmt_original: mc::cmt<'tcx>
}

impl<'a, 'tcx> GuaranteeLifetimeContext<'a, 'tcx> {

    fn check(&self, cmt: &mc::cmt<'tcx>, discr_scope: Option<ast::NodeId>) -> R {
        //! Main routine. Walks down `cmt` until we find the
        //! "guarantor".  Reports an error if `self.loan_region` is
        //! larger than scope of `cmt`.
        debug!("guarantee_lifetime.check(cmt={:?}, loan_region={:?})",
               cmt,
               self.loan_region);

        match cmt.cat {
            Categorization::Rvalue(..) |
            Categorization::Local(..) |                         // L-Local
            Categorization::Upvar(..) |
            Categorization::Deref(.., mc::BorrowedPtr(..)) |  // L-Deref-Borrowed
            Categorization::Deref(.., mc::Implicit(..)) |
            Categorization::Deref(.., mc::UnsafePtr(..)) => {
                self.check_scope(self.scope(cmt))
            }

            Categorization::StaticItem => {
                Ok(())
            }

            Categorization::Downcast(ref base, _) |
            Categorization::Deref(ref base, _, mc::Unique) |     // L-Deref-Send
            Categorization::Interior(ref base, _) => {             // L-Field
                self.check(base, discr_scope)
            }
        }
    }

    fn check_scope(&self, max_scope: &'tcx ty::Region) -> R {
        //! Reports an error if `loan_region` is larger than `max_scope`

        if !self.bccx.is_subregion_of(self.loan_region, max_scope) {
            Err(self.report_error(err_out_of_scope(max_scope, self.loan_region, self.cause)))
        } else {
            Ok(())
        }
    }

    fn scope(&self, cmt: &mc::cmt<'tcx>) -> &'tcx ty::Region {
        //! Returns the maximal region scope for the which the
        //! lvalue `cmt` is guaranteed to be valid without any
        //! rooting etc, and presuming `cmt` is not mutated.

        match cmt.cat {
            Categorization::Rvalue(temp_scope) => {
                temp_scope
            }
            Categorization::Upvar(..) => {
                self.bccx.tcx.mk_region(ty::ReScope(self.item_scope))
            }
            Categorization::Local(local_id) => {
                self.bccx.tcx.mk_region(ty::ReScope(
                    self.bccx.tcx.region_maps.var_scope(local_id)))
            }
            Categorization::StaticItem |
            Categorization::Deref(.., mc::UnsafePtr(..)) => {
                self.bccx.tcx.mk_region(ty::ReStatic)
            }
            Categorization::Deref(.., mc::BorrowedPtr(_, r)) |
            Categorization::Deref(.., mc::Implicit(_, r)) => {
                r
            }
            Categorization::Downcast(ref cmt, _) |
            Categorization::Deref(ref cmt, _, mc::Unique) |
            Categorization::Interior(ref cmt, _) => {
                self.scope(cmt)
            }
        }
    }

    fn report_error(&self, code: bckerr_code<'tcx>) {
        self.bccx.report(BckError { cmt: self.cmt_original.clone(),
                                    span: self.span,
                                    cause: BorrowViolation(self.cause),
                                    code: code });
    }
}
