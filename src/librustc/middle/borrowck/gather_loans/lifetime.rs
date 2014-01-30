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
use mc = middle::mem_categorization;
use middle::ty;
use util::ppaux::Repr;
use syntax::ast;
use syntax::codemap::Span;

type R = Result<(),()>;

pub fn guarantee_lifetime(bccx: &BorrowckCtxt,
                          item_scope_id: ast::NodeId,
                          root_scope_id: ast::NodeId,
                          span: Span,
                          cmt: mc::cmt,
                          loan_region: ty::Region,
                          loan_mutbl: LoanMutability) -> R {
    debug!("guarantee_lifetime(cmt={}, loan_region={})",
           cmt.repr(bccx.tcx), loan_region.repr(bccx.tcx));
    let ctxt = GuaranteeLifetimeContext {bccx: bccx,
                                         item_scope_id: item_scope_id,
                                         span: span,
                                         loan_region: loan_region,
                                         loan_mutbl: loan_mutbl,
                                         cmt_original: cmt,
                                         root_scope_id: root_scope_id};
    ctxt.check(cmt, None)
}

///////////////////////////////////////////////////////////////////////////
// Private

struct GuaranteeLifetimeContext<'a> {
    bccx: &'a BorrowckCtxt,

    // the node id of the function body for the enclosing item
    item_scope_id: ast::NodeId,

    // the node id of the innermost loop / function body; this is the
    // longest scope for which we can root managed boxes
    root_scope_id: ast::NodeId,

    span: Span,
    loan_region: ty::Region,
    loan_mutbl: LoanMutability,
    cmt_original: mc::cmt
}

impl<'a> GuaranteeLifetimeContext<'a> {
    fn tcx(&self) -> ty::ctxt {
        self.bccx.tcx
    }

    fn check(&self, cmt: mc::cmt, discr_scope: Option<ast::NodeId>) -> R {
        //! Main routine. Walks down `cmt` until we find the "guarantor".
        debug!("guarantee_lifetime.check(cmt={}, loan_region={})",
               cmt.repr(self.bccx.tcx),
               self.loan_region.repr(self.bccx.tcx));

        match cmt.cat {
            mc::cat_rvalue(..) |
            mc::cat_copied_upvar(..) |                  // L-Local
            mc::cat_local(..) |                         // L-Local
            mc::cat_arg(..) |                           // L-Local
            mc::cat_deref(_, _, mc::region_ptr(..)) |   // L-Deref-Borrowed
            mc::cat_deref(_, _, mc::unsafe_ptr(..)) => {
                let scope = self.scope(cmt);
                self.check_scope(scope)
            }

            mc::cat_stack_upvar(cmt) => {
                self.check(cmt, discr_scope)
            }

            mc::cat_static_item => {
                Ok(())
            }

            mc::cat_deref(base, derefs, mc::gc_ptr) => {
                let base_scope = self.scope(base);

                // L-Deref-Managed-Imm-User-Root
                let omit_root = (
                    self.bccx.is_subregion_of(self.loan_region, base_scope) &&
                    self.is_rvalue_or_immutable(base) &&
                    !self.is_moved(base)
                );

                if !omit_root {
                    // L-Deref-Managed-Imm-Compiler-Root
                    // L-Deref-Managed-Mut-Compiler-Root
                    self.check_root(cmt, base, derefs, discr_scope)
                } else {
                    debug!("omitting root, base={}, base_scope={:?}",
                           base.repr(self.tcx()), base_scope);
                    Ok(())
                }
            }

            mc::cat_downcast(base) |
            mc::cat_deref(base, _, mc::uniq_ptr) |     // L-Deref-Send
            mc::cat_interior(base, _) => {             // L-Field
                self.check(base, discr_scope)
            }

            mc::cat_discr(base, new_discr_scope) => {
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

    fn is_rvalue_or_immutable(&self,
                              cmt: mc::cmt) -> bool {
        //! We can omit the root on an `@T` value if the location
        //! that holds the box is either (1) an rvalue, in which case
        //! it is in a non-user-accessible temporary, or (2) an immutable
        //! lvalue.

        cmt.mutbl.is_immutable() || match cmt.guarantor().cat {
            mc::cat_rvalue(..) => true,
            _ => false
        }
    }

    fn check_root(&self,
                  cmt_deref: mc::cmt,
                  cmt_base: mc::cmt,
                  derefs: uint,
                  discr_scope: Option<ast::NodeId>) -> R {
        debug!("check_root(cmt_deref={}, cmt_base={}, derefs={:?}, \
                discr_scope={:?})",
               cmt_deref.repr(self.tcx()),
               cmt_base.repr(self.tcx()),
               derefs,
               discr_scope);

        // Make sure that the loan does not exceed the maximum time
        // that we can root the value, dynamically.
        let root_region = ty::ReScope(self.root_scope_id);
        if !self.bccx.is_subregion_of(self.loan_region, root_region) {
            return Err(self.report_error(
                err_out_of_root_scope(root_region, self.loan_region)));
        }

        // Extract the scope id that indicates how long the rooting is required
        let root_scope = match self.loan_region {
            ty::ReScope(id) => id,
            _ => {
                // the check above should fail for anything is not ReScope
                self.bccx.tcx.sess.span_bug(
                    cmt_base.span,
                    format!("Cannot issue root for scope region: {:?}",
                         self.loan_region));
            }
        };

        // If inside of a match arm, expand the rooting to the entire
        // match. See the detailed discussion in `check()` above.
        let root_scope = match discr_scope {
            None => root_scope,
            Some(id) => {
                if self.bccx.is_subscope_of(root_scope, id) {
                    id
                } else {
                    root_scope
                }
            }
        };

        // Add a record of what is required
        let rm_key = root_map_key {id: cmt_deref.id, derefs: derefs};
        let root_info = RootInfo {scope: root_scope};

        let mut root_map = self.bccx.root_map.borrow_mut();
        root_map.get().insert(rm_key, root_info);

        debug!("root_key: {:?} root_info: {:?}", rm_key, root_info);
        Ok(())
    }

    fn check_scope(&self, max_scope: ty::Region) -> R {
        //! Reports an error if `loan_region` is larger than `valid_scope`

        if !self.bccx.is_subregion_of(self.loan_region, max_scope) {
            Err(self.report_error(err_out_of_scope(max_scope, self.loan_region)))
        } else {
            Ok(())
        }
    }

    fn is_moved(&self, cmt: mc::cmt) -> bool {
        //! True if `cmt` is something that is potentially moved
        //! out of the current stack frame.

        match cmt.guarantor().cat {
            mc::cat_local(id) |
            mc::cat_arg(id) => {
                let moved_variables_set = self.bccx
                                              .moved_variables_set
                                              .borrow();
                moved_variables_set.get().contains(&id)
            }
            mc::cat_rvalue(..) |
            mc::cat_static_item |
            mc::cat_copied_upvar(..) |
            mc::cat_deref(..) => {
                false
            }
            r @ mc::cat_downcast(..) |
            r @ mc::cat_interior(..) |
            r @ mc::cat_stack_upvar(..) |
            r @ mc::cat_discr(..) => {
                self.tcx().sess.span_bug(
                    cmt.span,
                    format!("illegal guarantor category: {:?}", r));
            }
        }
    }

    fn scope(&self, cmt: mc::cmt) -> ty::Region {
        //! Returns the maximal region scope for the which the
        //! lvalue `cmt` is guaranteed to be valid without any
        //! rooting etc, and presuming `cmt` is not mutated.

        // See the SCOPE(LV) function in doc.rs

        match cmt.cat {
            mc::cat_rvalue(temp_scope) => {
                temp_scope
            }
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
            mc::cat_deref(_, _, mc::unsafe_ptr(..)) => {
                ty::ReStatic
            }
            mc::cat_deref(_, _, mc::region_ptr(_, r)) => {
                r
            }
            mc::cat_downcast(cmt) |
            mc::cat_deref(cmt, _, mc::uniq_ptr) |
            mc::cat_deref(cmt, _, mc::gc_ptr) |
            mc::cat_interior(cmt, _) |
            mc::cat_stack_upvar(cmt) |
            mc::cat_discr(cmt, _) => {
                self.scope(cmt)
            }
        }
    }

    fn report_error(&self, code: bckerr_code) {
        self.bccx.report(BckError {
            cmt: self.cmt_original,
            span: self.span,
            code: code
        });
    }
}
