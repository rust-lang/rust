// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Computes the restrictions that result from a borrow.
 */

use std::vec_ng::Vec;
use std::vec_ng;
use middle::borrowck::*;
use mc = middle::mem_categorization;
use middle::ty;
use syntax::codemap::Span;
use util::ppaux::Repr;

pub enum RestrictionResult {
    Safe,
    SafeIf(@LoanPath, Vec<Restriction> )
}

pub fn compute_restrictions(bccx: &BorrowckCtxt,
                            span: Span,
                            cause: LoanCause,
                            cmt: mc::cmt,
                            loan_region: ty::Region,
                            restr: RestrictionSet) -> RestrictionResult {
    let ctxt = RestrictionsContext {
        bccx: bccx,
        span: span,
        cause: cause,
        cmt_original: cmt,
        loan_region: loan_region,
    };

    ctxt.restrict(cmt, restr)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct RestrictionsContext<'a> {
    bccx: &'a BorrowckCtxt,
    span: Span,
    cmt_original: mc::cmt,
    loan_region: ty::Region,
    cause: LoanCause,
}

impl<'a> RestrictionsContext<'a> {
    fn restrict(&self,
                cmt: mc::cmt,
                restrictions: RestrictionSet) -> RestrictionResult {
        debug!("restrict(cmt={}, restrictions={})",
               cmt.repr(self.bccx.tcx),
               restrictions.repr(self.bccx.tcx));

        match cmt.cat {
            mc::cat_rvalue(..) => {
                // Effectively, rvalues are stored into a
                // non-aliasable temporary on the stack. Since they
                // are inherently non-aliasable, they can only be
                // accessed later through the borrow itself and hence
                // must inherently comply with its terms.
                Safe
            }

            mc::cat_local(local_id) |
            mc::cat_arg(local_id) |
            mc::cat_upvar(ty::UpvarId {var_id: local_id, ..}, _) => {
                // R-Variable
                let lp = @LpVar(local_id);
                SafeIf(lp, vec!(Restriction {loan_path: lp,
                                          set: restrictions}))
            }

            mc::cat_downcast(cmt_base) => {
                // When we borrow the interior of an enum, we have to
                // ensure the enum itself is not mutated, because that
                // could cause the type of the memory to change.
                self.restrict(
                    cmt_base,
                    restrictions | RESTR_MUTATE)
            }

            mc::cat_interior(cmt_base, i) => {
                // R-Field
                //
                // Overwriting the base would not change the type of
                // the memory, so no additional restrictions are
                // needed.
                let result = self.restrict(cmt_base, restrictions);
                self.extend(result, cmt.mutbl, LpInterior(i), restrictions)
            }

            mc::cat_deref(cmt_base, _, pk @ mc::OwnedPtr) => {
                // R-Deref-Send-Pointer
                //
                // When we borrow the interior of an owned pointer, we
                // cannot permit the base to be mutated, because that
                // would cause the unique pointer to be freed.
                let result = self.restrict(
                    cmt_base,
                    restrictions | RESTR_MUTATE);
                self.extend(result, cmt.mutbl, LpDeref(pk), restrictions)
            }

            mc::cat_copied_upvar(..) | // FIXME(#2152) allow mutation of upvars
            mc::cat_static_item(..) => {
                Safe
            }

            mc::cat_deref(cmt_base, _, mc::BorrowedPtr(ty::ImmBorrow, lt)) |
            mc::cat_deref(cmt_base, _, mc::BorrowedPtr(ty::UniqueImmBorrow, lt)) => {
                // R-Deref-Imm-Borrowed
                if !self.bccx.is_subregion_of(self.loan_region, lt) {
                    self.bccx.report(
                        BckError {
                            span: self.span,
                            cause: self.cause,
                            cmt: cmt_base,
                            code: err_borrowed_pointer_too_short(
                                self.loan_region, lt, restrictions)});
                    return Safe;
                }
                Safe
            }

            mc::cat_deref(_, _, mc::GcPtr) => {
                // R-Deref-Imm-Managed
                Safe
            }

            mc::cat_deref(cmt_base, _, pk @ mc::BorrowedPtr(ty::MutBorrow, lt)) => {
                // R-Deref-Mut-Borrowed
                if !self.bccx.is_subregion_of(self.loan_region, lt) {
                    self.bccx.report(
                        BckError {
                            span: self.span,
                            cause: self.cause,
                            cmt: cmt_base,
                            code: err_borrowed_pointer_too_short(
                                self.loan_region, lt, restrictions)});
                    return Safe;
                }

                let result = self.restrict(cmt_base, restrictions);
                self.extend(result, cmt.mutbl, LpDeref(pk), restrictions)
            }

            mc::cat_deref(_, _, mc::UnsafePtr(..)) => {
                // We are very trusting when working with unsafe pointers.
                Safe
            }

            mc::cat_discr(cmt_base, _) => {
                self.restrict(cmt_base, restrictions)
            }
        }
    }

    fn extend(&self,
              result: RestrictionResult,
              mc: mc::MutabilityCategory,
              elem: LoanPathElem,
              restrictions: RestrictionSet) -> RestrictionResult {
        match result {
            Safe => Safe,
            SafeIf(base_lp, base_vec) => {
                let lp = @LpExtend(base_lp, mc, elem);
                SafeIf(lp, vec_ng::append_one(base_vec,
                                              Restriction {
                                                  loan_path: lp,
                                                  set: restrictions
                                              }))
            }
        }
    }
}
