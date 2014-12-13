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

pub use self::RestrictionResult::*;

use borrowck::*;
use borrowck::LoanPathElem::*;
use borrowck::LoanPathKind::*;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::ty;
use rustc::util::ppaux::Repr;
use syntax::codemap::Span;

use std::rc::Rc;

#[deriving(Show)]
pub enum RestrictionResult<'tcx> {
    Safe,
    SafeIf(Rc<LoanPath<'tcx>>, Vec<Rc<LoanPath<'tcx>>>)
}

pub fn compute_restrictions<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                      span: Span,
                                      cause: euv::LoanCause,
                                      cmt: mc::cmt<'tcx>,
                                      loan_region: ty::Region)
                                      -> RestrictionResult<'tcx> {
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
                cmt: mc::cmt<'tcx>) -> RestrictionResult<'tcx> {
        debug!("restrict(cmt={})", cmt.repr(self.bccx.tcx));

        let new_lp = |v: LoanPathKind<'tcx>| Rc::new(LoanPath::new(v, cmt.ty));

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
                let lp = new_lp(LpVar(local_id));
                SafeIf(lp.clone(), vec![lp])
            }

            mc::cat_upvar(mc::Upvar { id, .. }) => {
                // R-Variable, captured into closure
                let lp = new_lp(LpUpvar(id));
                SafeIf(lp.clone(), vec![lp])
            }

            mc::cat_downcast(cmt_base, _) => {
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
                self.extend(result, &cmt, LpInterior(i))
            }

            mc::cat_static_item(..) => {
                Safe
            }

            mc::cat_deref(cmt_base, _, pk) => {
                match pk {
                    mc::OwnedPtr => {
                        // R-Deref-Send-Pointer
                        //
                        // When we borrow the interior of an owned pointer, we
                        // cannot permit the base to be mutated, because that
                        // would cause the unique pointer to be freed.
                        //
                        // Eventually we should make these non-special and
                        // just rely on Deref<T> implementation.
                        let result = self.restrict(cmt_base);
                        self.extend(result, &cmt, LpDeref(pk))
                    }
                    mc::Implicit(bk, lt) | mc::BorrowedPtr(bk, lt) => {
                        // R-Deref-[Mut-]Borrowed
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

                        match bk {
                            ty::ImmBorrow => Safe,
                            ty::MutBorrow | ty::UniqueImmBorrow => {
                                // R-Deref-Mut-Borrowed
                                //
                                // The referent can be aliased after the
                                // references lifetime ends (by a newly-unfrozen
                                // borrow).
                                let result = self.restrict(cmt_base);
                                self.extend(result, &cmt, LpDeref(pk))
                            }
                        }
                    }
                    // Borrowck is not relevant for unsafe pointers
                    mc::UnsafePtr(..) => Safe
                }
            }
        }
    }

    fn extend(&self,
              result: RestrictionResult<'tcx>,
              cmt: &mc::cmt<'tcx>,
              elem: LoanPathElem) -> RestrictionResult<'tcx> {
        match result {
            Safe => Safe,
            SafeIf(base_lp, mut base_vec) => {
                let v = LpExtend(base_lp, cmt.mutbl, elem);
                let lp = Rc::new(LoanPath::new(v, cmt.ty));
                base_vec.push(lp.clone());
                SafeIf(lp, base_vec)
            }
        }
    }
}
