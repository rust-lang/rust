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

use std::vec;
use middle::borrowck::*;
use mc = middle::mem_categorization;
use middle::ty;
use syntax::ast::{MutImmutable, MutMutable};
use syntax::codemap::Span;

pub enum RestrictionResult {
    Safe,
    SafeIf(@LoanPath, ~[Restriction])
}

pub fn compute_restrictions(bccx: &BorrowckCtxt,
                            span: Span,
                            cmt: mc::cmt,
                            loan_region: ty::Region,
                            restr: RestrictionSet) -> RestrictionResult {
    let ctxt = RestrictionsContext {
        bccx: bccx,
        span: span,
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
}

impl<'a> RestrictionsContext<'a> {
    fn restrict(&self,
                cmt: mc::cmt,
                restrictions: RestrictionSet) -> RestrictionResult {

        // Check for those cases where we cannot control the aliasing
        // and make sure that we are not being asked to.
        match cmt.freely_aliasable() {
            None => {}
            Some(cause) => {
                self.check_aliasing_permitted(cause, restrictions);
            }
        }

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
            mc::cat_self(local_id) => {
                // R-Variable
                let lp = @LpVar(local_id);
                SafeIf(lp, ~[Restriction {loan_path: lp,
                                          set: restrictions}])
            }

            mc::cat_downcast(cmt_base) => {
                // When we borrow the interior of an enum, we have to
                // ensure the enum itself is not mutated, because that
                // could cause the type of the memory to change.
                self.restrict(
                    cmt_base,
                    restrictions | RESTR_MUTATE | RESTR_CLAIM)
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

            mc::cat_deref(cmt_base, _, pk @ mc::uniq_ptr) => {
                // R-Deref-Send-Pointer
                //
                // When we borrow the interior of an owned pointer, we
                // cannot permit the base to be mutated, because that
                // would cause the unique pointer to be freed.
                let result = self.restrict(
                    cmt_base,
                    restrictions | RESTR_MUTATE | RESTR_CLAIM);
                self.extend(result, cmt.mutbl, LpDeref(pk), restrictions)
            }

            mc::cat_copied_upvar(..) | // FIXME(#2152) allow mutation of upvars
            mc::cat_static_item(..) => {
                Safe
            }

            mc::cat_deref(cmt_base, _, mc::region_ptr(MutImmutable, lt)) => {
                // R-Deref-Imm-Borrowed
                if !self.bccx.is_subregion_of(self.loan_region, lt) {
                    self.bccx.report(
                        BckError {
                            span: self.span,
                            cmt: cmt_base,
                            code: err_borrowed_pointer_too_short(
                                self.loan_region, lt, restrictions)});
                    return Safe;
                }
                Safe
            }

            mc::cat_deref(_, _, mc::gc_ptr) => {
                // R-Deref-Imm-Managed
                Safe
            }

            mc::cat_deref(cmt_base, _, pk @ mc::region_ptr(MutMutable, lt)) => {
                // R-Deref-Mut-Borrowed
                if !self.bccx.is_subregion_of(self.loan_region, lt) {
                    self.bccx.report(
                        BckError {
                            span: self.span,
                            cmt: cmt_base,
                            code: err_borrowed_pointer_too_short(
                                self.loan_region, lt, restrictions)});
                    return Safe;
                }

                let result = self.restrict(cmt_base, restrictions);
                self.extend(result, cmt.mutbl, LpDeref(pk), restrictions)
            }

            mc::cat_deref(_, _, mc::unsafe_ptr(..)) => {
                // We are very trusting when working with unsafe pointers.
                Safe
            }

            mc::cat_stack_upvar(cmt_base) |
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
                SafeIf(lp, vec::append_one(base_vec,
                                           Restriction {loan_path: lp,
                                                        set: restrictions}))
            }
        }
    }

    fn check_aliasing_permitted(&self,
                                cause: mc::AliasableReason,
                                restrictions: RestrictionSet) {
        //! This method is invoked when the current `cmt` is something
        //! where aliasing cannot be controlled. It reports an error if
        //! the restrictions required that it not be aliased; currently
        //! this only occurs when re-borrowing an `&mut` pointer.
        //!
        //! NB: To be 100% consistent, we should report an error if
        //! RESTR_FREEZE is found, because we cannot prevent freezing,
        //! nor would we want to. However, we do not report such an
        //! error, because this restriction only occurs when the user
        //! is creating an `&mut` pointer to immutable or read-only
        //! data, and there is already another piece of code that
        //! checks for this condition.

        if restrictions.intersects(RESTR_ALIAS) {
            self.bccx.report_aliasability_violation(
                self.span,
                BorrowViolation,
                cause);
        }
    }
}
