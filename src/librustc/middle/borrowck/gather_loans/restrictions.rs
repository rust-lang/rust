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

use middle::borrowck::*;
use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty;
use syntax::codemap::Span;
use util::ppaux::Repr;

use std::rc::Rc;

pub enum RestrictionResult {
    Safe,
    SafeIf(Rc<LoanPath>, Vec<Rc<LoanPath>>)
}

pub fn compute_restrictions(bccx: &BorrowckCtxt,
                            span: Span,
                            cause: euv::LoanCause,
                            cmt: mc::cmt,
                            loan_region: ty::Region) -> RestrictionResult {
    let ctxt = RestrictionsContext {
        bccx: bccx,
        span: span,
        cause: cause,
        loan_region: loan_region,
    };

    ctxt.restrict(cmt)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct RestrictionsContext<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    span: Span,
    loan_region: ty::Region,
    cause: euv::LoanCause,
}

impl<'a, 'tcx> RestrictionsContext<'a, 'tcx> {
    fn restrict(&self,
                cmt: mc::cmt) -> RestrictionResult {
        debug!("restrict(cmt={})", cmt.repr(self.bccx.tcx));

        match cmt.cat.clone() {
            mc::cat_rvalue(..) => {
                // Effectively, rvalues are stored into a
                // non-aliasable temporary on the stack. Since they
                // are inherently non-aliasable, they can only be
                // accessed later through the borrow itself and hence
                // must inherently comply with its terms.
                Safe
            }

            mc::cat_local(local_id) => {
                // R-Variable, locally declared
                let lp = Rc::new(LpVar(local_id));
                SafeIf(lp.clone(), vec![lp])
            }

            mc::cat_upvar(upvar_id, _) => {
                // R-Variable, captured into closure
                let lp = Rc::new(LpUpvar(upvar_id));
                SafeIf(lp.clone(), vec![lp])
            }

            mc::cat_copied_upvar(mc::CopiedUpvar { upvar_id, .. }) => {
                // R-Variable, copied/moved into closure
                let lp = Rc::new(LpVar(upvar_id));
                SafeIf(lp.clone(), vec![lp])
            }

            mc::cat_downcast(cmt_base) => {
                // When we borrow the interior of an enum, we have to
                // ensure the enum itself is not mutated, because that
                // could cause the type of the memory to change.
                self.restrict(cmt_base)
            }

            mc::cat_interior(cmt_base, i) => {
                // R-Field
                //
                // Overwriting the base would not change the type of
                // the memory, so no additional restrictions are
                // needed.
                let result = self.restrict(cmt_base);
                self.extend(result, cmt.mutbl, LpInterior(i))
            }

            mc::cat_deref(cmt_base, _, pk @ mc::OwnedPtr) |
            mc::cat_deref(cmt_base, _, pk @ mc::GcPtr) => {
                // R-Deref-Send-Pointer
                //
                // When we borrow the interior of an owned pointer, we
                // cannot permit the base to be mutated, because that
                // would cause the unique pointer to be freed.
                //
                // For a managed pointer, the rules are basically the
                // same, because this could be the last ref.
                // Eventually we should make these non-special and
                // just rely on Deref<T> implementation.
                let result = self.restrict(cmt_base);
                self.extend(result, cmt.mutbl, LpDeref(pk))
            }

            mc::cat_static_item(..) => {
                Safe
            }

            mc::cat_deref(cmt_base, _, mc::BorrowedPtr(ty::ImmBorrow, lt)) |
            mc::cat_deref(cmt_base, _, mc::Implicit(ty::ImmBorrow, lt)) => {
                // R-Deref-Imm-Borrowed
                if !self.bccx.is_subregion_of(self.loan_region, lt) {
                    self.bccx.report(
                        BckError {
                            span: self.span,
                            cause: self.cause,
                            cmt: cmt_base,
                            code: err_borrowed_pointer_too_short(
                                self.loan_region, lt)});
                    return Safe;
                }
                Safe
            }

            mc::cat_deref(cmt_base, _, pk) => {
                match pk {
                    mc::BorrowedPtr(ty::MutBorrow, lt) |
                    mc::BorrowedPtr(ty::UniqueImmBorrow, lt) |
                    mc::Implicit(ty::MutBorrow, lt) |
                    mc::Implicit(ty::UniqueImmBorrow, lt) => {
                        // R-Deref-Mut-Borrowed
                        if !self.bccx.is_subregion_of(self.loan_region, lt) {
                            self.bccx.report(
                                BckError {
                                    span: self.span,
                                    cause: self.cause,
                                    cmt: cmt_base,
                                    code: err_borrowed_pointer_too_short(
                                        self.loan_region, lt)});
                            return Safe;
                        }

                        let result = self.restrict(cmt_base);
                        self.extend(result, cmt.mutbl, LpDeref(pk))
                    }
                    mc::UnsafePtr(..) => {
                        // We are very trusting when working with unsafe
                        // pointers.
                        Safe
                    }
                    _ => {
                        self.bccx.tcx.sess.span_bug(self.span,
                                                    "unhandled memcat in \
                                                     cat_deref")
                    }
                }
            }

            mc::cat_discr(cmt_base, _) => {
                self.restrict(cmt_base)
            }
        }
    }

    fn extend(&self,
              result: RestrictionResult,
              mc: mc::MutabilityCategory,
              elem: LoanPathElem) -> RestrictionResult {
        match result {
            Safe => Safe,
            SafeIf(base_lp, mut base_vec) => {
                let lp = Rc::new(LpExtend(base_lp, mc, elem));
                base_vec.push(lp.clone());
                SafeIf(lp, base_vec)
            }
        }
    }
}
