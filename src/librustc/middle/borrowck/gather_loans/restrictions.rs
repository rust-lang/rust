// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Computes the restrictions that result from a borrow.


use std::vec;
use middle::borrowck::*;
use mc = middle::mem_categorization;
use middle::ty;
use syntax::ast::{m_imm, m_mutbl};
use syntax::codemap::span;

pub enum RestrictionResult {
    Safe,
    SafeIf(@LoanPath, ~[Restriction])
}

pub fn compute_restrictions(bccx: @BorrowckCtxt,
                            span: span,
                            cmt: mc::cmt,
                            restr: RestrictionSet) -> RestrictionResult {
    let ctxt = RestrictionsContext {
        bccx: bccx,
        span: span,
        cmt_original: cmt
    };

    ctxt.restrict(cmt, restr)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct RestrictionsContext {
    bccx: @BorrowckCtxt,
    span: span,
    cmt_original: mc::cmt
}

impl RestrictionsContext {
    fn tcx(&self) -> ty::ctxt {
        self.bccx.tcx
    }

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
            mc::cat_rvalue(*) => {
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

            mc::cat_copied_upvar(*) | // FIXME(#2152) allow mutation of upvars
            mc::cat_static_item(*) |
            mc::cat_deref(_, _, mc::region_ptr(m_imm, _)) |
            mc::cat_deref(_, _, mc::gc_ptr(m_imm)) => {
                // R-Deref-Imm-Borrowed
                Safe
            }

            mc::cat_deref(cmt_base, _, pk @ mc::gc_ptr(m_mutbl)) => {
                // R-Deref-Managed-Borrowed
                //
                // Technically, no restrictions are *necessary* here.
                // The validity of the borrow is guaranteed
                // dynamically.  However, nonetheless we add a
                // restriction to make a "best effort" to report
                // static errors. For example, if there is code like
                //
                //    let v = @mut ~[1, 2, 3];
                //    for e in v.iter() {
                //        v.push(e + 1);
                //    }
                //
                // Then the code below would add restrictions on `*v`,
                // which means that an error would be reported
                // here. This of course is not perfect. For example,
                // a function like the following would not report an error
                // at compile-time but would fail dynamically:
                //
                //    let v = @mut ~[1, 2, 3];
                //    let w = v;
                //    for e in v.iter() {
                //        w.push(e + 1);
                //    }
                //
                // In addition, we only add a restriction for those cases
                // where we can construct a sensible loan path, so an
                // example like the following will fail dynamically:
                //
                //    impl V {
                //      fn get_list(&self) -> @mut ~[int];
                //    }
                //    ...
                //    let v: &V = ...;
                //    for e in v.get_list().iter() {
                //        v.get_list().push(e + 1);
                //    }
                match opt_loan_path(cmt_base) {
                    None => Safe,
                    Some(lp_base) => {
                        let lp = @LpExtend(lp_base, cmt.mutbl, LpDeref(pk));
                        SafeIf(lp, ~[Restriction {loan_path: lp,
                                                  set: restrictions}])
                    }
                }
            }

            mc::cat_deref(cmt_base, _, pk @ mc::region_ptr(m_mutbl, _)) => {
                // Because an `&mut` pointer does not inherit its
                // mutability, we can only prevent mutation or prevent
                // freezing if it is not aliased. Therefore, in such
                // cases we restrict aliasing on `cmt_base`.
                if restrictions != RESTR_EMPTY {
                    // R-Deref-Mut-Borrowed-1
                    let result = self.restrict(
                        cmt_base,
                        RESTR_ALIAS | RESTR_MUTATE | RESTR_CLAIM);
                    self.extend(result, cmt.mutbl, LpDeref(pk), restrictions)
                } else {
                    // R-Deref-Mut-Borrowed-2
                    Safe
                }
            }

            mc::cat_deref(_, _, mc::unsafe_ptr(*)) => {
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

    fn check_no_mutability_control(&self,
                                   cmt: mc::cmt,
                                   restrictions: RestrictionSet) {
        if restrictions.intersects(RESTR_MUTATE | RESTR_FREEZE | RESTR_CLAIM) {
            self.bccx.report(BckError {span: self.span,
                                       cmt: cmt,
                                       code: err_freeze_aliasable_const});
        }
    }
}
