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
 * This module implements the check that the lifetime of a borrow
 * does not exceed the lifetime of the value being borrowed.
 */

use middle::borrowck::*;
use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty;
use util::ppaux::Repr;
use syntax::ast;
use syntax::codemap::Span;

type R = Result<(),()>;

pub fn guarantee_lifetime(bccx: &BorrowckCtxt,
                          item_scope_id: ast::NodeId,
                          span: Span,
                          cause: euv::LoanCause,
                          cmt: mc::cmt,
                          loan_region: ty::Region,
                          _: ty::BorrowKind)
                          -> Result<(),()> {
    debug!("guarantee_lifetime(cmt={}, loan_region={})",
           cmt.repr(bccx.tcx), loan_region.repr(bccx.tcx));
    let ctxt = GuaranteeLifetimeContext {bccx: bccx,
                                         item_scope_id: item_scope_id,
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

    // the node id of the function body for the enclosing item
    item_scope_id: ast::NodeId,

    span: Span,
    cause: euv::LoanCause,
    loan_region: ty::Region,
    cmt_original: mc::cmt
}

impl<'a, 'tcx> GuaranteeLifetimeContext<'a, 'tcx> {

    fn check(&self, cmt: &mc::cmt, discr_scope: Option<ast::NodeId>) -> R {
        //! Main routine. Walks down `cmt` until we find the "guarantor".
        debug!("guarantee_lifetime.check(cmt={}, loan_region={})",
               cmt.repr(self.bccx.tcx),
               self.loan_region.repr(self.bccx.tcx));

        match cmt.cat {
            mc::cat_rvalue(..) |
            mc::cat_copied_upvar(..) |                  // L-Local
            mc::cat_local(..) |                         // L-Local
            mc::cat_arg(..) |                           // L-Local
            mc::cat_upvar(..) |
            mc::cat_deref(_, _, mc::BorrowedPtr(..)) |  // L-Deref-Borrowed
            mc::cat_deref(_, _, mc::Implicit(..)) |
            mc::cat_deref(_, _, mc::UnsafePtr(..)) => {
                self.check_scope(self.scope(cmt))
            }

            mc::cat_static_item => {
                Ok(())
            }

            mc::cat_downcast(ref base) |
            mc::cat_deref(ref base, _, mc::OwnedPtr) |     // L-Deref-Send
            mc::cat_interior(ref base, _) |                // L-Field
            mc::cat_deref(ref base, _, mc::GcPtr) => {
                self.check(base, discr_scope)
            }

            mc::cat_discr(ref base, new_discr_scope) => {
                // Subtle: in a match, we must ensure that each binding
                // variable remains valid for the duration of the arm in
                // which it appears, presuming that this arm is taken.
                // But it is inconvenient in trans to root something just
                // for one arm.  Therefore, we insert a cat_discr(),
                // basically a special kind of category that says "if this
                // value must be dynamically rooted, root it for the scope
                // `match_id`".
                //
                // As an example, consider this scenario:
                //
                //    let mut x = @Some(3);
                //    match *x { Some(y) {...} None {...} }
                //
                // Technically, the value `x` need only be rooted
                // in the `some` arm.  However, we evaluate `x` in trans
                // before we know what arm will be taken, so we just
                // always root it for the duration of the match.
                //
                // As a second example, consider *this* scenario:
                //
                //    let x = @@Some(3);
                //    match x { @@Some(y) {...} @@None {...} }
                //
                // Here again, `x` need only be rooted in the `some` arm.
                // In this case, the value which needs to be rooted is
                // found only when checking which pattern matches: but
                // this check is done before entering the arm.  Therefore,
                // even in this case we just choose to keep the value
                // rooted for the entire match.  This means the value will be
                // rooted even if the none arm is taken.  Oh well.
                //
                // At first, I tried to optimize the second case to only
                // root in one arm, but the result was suboptimal: first,
                // it interfered with the construction of phi nodes in the
                // arm, as we were adding code to root values before the
                // phi nodes were added.  This could have been addressed
                // with a second basic block.  However, the naive approach
                // also yielded suboptimal results for patterns like:
                //
                //    let x = @@...;
                //    match x { @@some_variant(y) | @@some_other_variant(y) =>
                //
                // The reason is that we would root the value once for
                // each pattern and not once per arm.  This is also easily
                // fixed, but it's yet more code for what is really quite
                // the corner case.
                //
                // Nonetheless, if you decide to optimize this case in the
                // future, you need only adjust where the cat_discr()
                // node appears to draw the line between what will be rooted
                // in the *arm* vs the *match*.
                self.check(base, Some(new_discr_scope))
            }
        }
    }

    fn check_scope(&self, max_scope: ty::Region) -> R {
        //! Reports an error if `loan_region` is larger than `valid_scope`

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
            mc::cat_upvar(..) |
            mc::cat_copied_upvar(_) => {
                ty::ReScope(self.item_scope_id)
            }
            mc::cat_static_item => {
                ty::ReStatic
            }
            mc::cat_local(local_id) |
            mc::cat_arg(local_id) => {
                ty::ReScope(self.bccx.tcx.region_maps.var_scope(local_id))
            }
            mc::cat_deref(_, _, mc::UnsafePtr(..)) => {
                ty::ReStatic
            }
            mc::cat_deref(_, _, mc::BorrowedPtr(_, r)) |
            mc::cat_deref(_, _, mc::Implicit(_, r)) => {
                r
            }
            mc::cat_downcast(ref cmt) |
            mc::cat_deref(ref cmt, _, mc::OwnedPtr) |
            mc::cat_deref(ref cmt, _, mc::GcPtr) |
            mc::cat_interior(ref cmt, _) |
            mc::cat_discr(ref cmt, _) => {
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
