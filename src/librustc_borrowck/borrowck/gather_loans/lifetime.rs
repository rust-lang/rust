//! This module implements the check that the lifetime of a borrow
//! does not exceed the lifetime of the value being borrowed.

use crate::borrowck::*;
use rustc::hir::HirId;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::ty;

use syntax_pos::Span;
use log::debug;

type R = Result<(),()>;

pub fn guarantee_lifetime<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                    item_scope: region::Scope,
                                    span: Span,
                                    cause: euv::LoanCause,
                                    cmt: &'a mc::cmt_<'tcx>,
                                    loan_region: ty::Region<'tcx>)
                                    -> Result<(),()> {
    //! Reports error if `loan_region` is larger than S
    //! where S is `item_scope` if `cmt` is an upvar,
    //! and is scope of `cmt` otherwise.
    debug!("guarantee_lifetime(cmt={:?}, loan_region={:?})",
           cmt, loan_region);
    let ctxt = GuaranteeLifetimeContext {bccx: bccx,
                                         item_scope,
                                         span,
                                         cause,
                                         loan_region,
                                         cmt_original: cmt};
    ctxt.check(cmt, None)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct GuaranteeLifetimeContext<'a, 'tcx> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,

    // the scope of the function body for the enclosing item
    item_scope: region::Scope,

    span: Span,
    cause: euv::LoanCause,
    loan_region: ty::Region<'tcx>,
    cmt_original: &'a mc::cmt_<'tcx>
}

impl<'a, 'tcx> GuaranteeLifetimeContext<'a, 'tcx> {
    fn check(&self, cmt: &mc::cmt_<'tcx>, discr_scope: Option<HirId>) -> R {
        //! Main routine. Walks down `cmt` until we find the
        //! "guarantor". Reports an error if `self.loan_region` is
        //! larger than scope of `cmt`.
        debug!("guarantee_lifetime.check(cmt={:?}, loan_region={:?})",
               cmt,
               self.loan_region);

        match cmt.cat {
            Categorization::Rvalue(..) |
            Categorization::ThreadLocal(..) |
            Categorization::Local(..) |                     // L-Local
            Categorization::Upvar(..) |
            Categorization::Deref(_, mc::BorrowedPtr(..)) | // L-Deref-Borrowed
            Categorization::Deref(_, mc::UnsafePtr(..)) => {
                self.check_scope(self.scope(cmt))
            }

            Categorization::StaticItem => {
                Ok(())
            }

            Categorization::Downcast(ref base, _) |
            Categorization::Deref(ref base, mc::Unique) |   // L-Deref-Send
            Categorization::Interior(ref base, _) => {      // L-Field
                self.check(base, discr_scope)
            }
        }
    }

    fn check_scope(&self, max_scope: ty::Region<'tcx>) -> R {
        //! Reports an error if `loan_region` is larger than `max_scope`

        if !self.bccx.is_subregion_of(self.loan_region, max_scope) {
            Err(self.report_error(err_out_of_scope(max_scope, self.loan_region, self.cause)))
        } else {
            Ok(())
        }
    }

    fn scope(&self, cmt: &mc::cmt_<'tcx>) -> ty::Region<'tcx> {
        //! Returns the maximal region scope for the which the
        //! place `cmt` is guaranteed to be valid without any
        //! rooting etc, and presuming `cmt` is not mutated.

        match cmt.cat {
            Categorization::ThreadLocal(temp_scope) |
            Categorization::Rvalue(temp_scope) => {
                temp_scope
            }
            Categorization::Upvar(..) => {
                self.bccx.tcx.mk_region(ty::ReScope(self.item_scope))
            }
            Categorization::Local(hir_id) => {
                self.bccx.tcx.mk_region(ty::ReScope(
                    self.bccx.region_scope_tree.var_scope(hir_id.local_id)))
            }
            Categorization::StaticItem |
            Categorization::Deref(_, mc::UnsafePtr(..)) => {
                self.bccx.tcx.lifetimes.re_static
            }
            Categorization::Deref(_, mc::BorrowedPtr(_, r)) => {
                r
            }
            Categorization::Downcast(ref cmt, _) |
            Categorization::Deref(ref cmt, mc::Unique) |
            Categorization::Interior(ref cmt, _) => {
                self.scope(cmt)
            }
        }
    }

    fn report_error(&self, code: bckerr_code<'tcx>) {
        self.bccx.report(BckError { cmt: self.cmt_original,
                                    span: self.span,
                                    cause: BorrowViolation(self.cause),
                                    code: code });
    }
}
