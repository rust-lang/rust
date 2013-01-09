// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ----------------------------------------------------------------------
// Loan(Ex, M, S) = Ls holds if ToAddr(Ex) will remain valid for the entirety
// of the scope S, presuming that the returned set of loans `Ls` are honored.

use core::prelude::*;

use middle::borrowck::{Loan, bckres, borrowck_ctxt, cmt, err_mutbl};
use middle::borrowck::{err_out_of_scope};
use middle::mem_categorization::{cat_arg, cat_binding, cat_discr, cat_comp};
use middle::mem_categorization::{cat_deref, cat_discr, cat_local};
use middle::mem_categorization::{cat_special, cat_stack_upvar, comp_field};
use middle::mem_categorization::{comp_index, comp_variant, region_ptr};
use middle::ty;
use util::common::indenter;

use core::result::{Err, Ok, Result};
use syntax::ast::{m_const, m_imm, m_mutbl};
use syntax::ast;

export public_methods;

impl borrowck_ctxt {
    fn loan(cmt: cmt,
            scope_region: ty::Region,
            mutbl: ast::mutability) -> bckres<~[Loan]> {
        let lc = LoanContext {
            bccx: self,
            scope_region: scope_region,
            loans: ~[]
        };
        match lc.loan(cmt, mutbl) {
          Err(ref e) => Err((*e)),
          Ok(()) => {
              let LoanContext {loans, _} = move lc;
              Ok(loans)
          }
        }
    }
}

struct LoanContext {
    bccx: borrowck_ctxt,

    // the region scope for which we must preserve the memory
    scope_region: ty::Region,

    // accumulated list of loans that will be required
    mut loans: ~[Loan]
}

impl LoanContext {
    fn tcx(&self) -> ty::ctxt { self.bccx.tcx }

    fn issue_loan(&self,
                  cmt: cmt,
                  scope_ub: ty::Region,
                  req_mutbl: ast::mutability) -> bckres<()> {
        if self.bccx.is_subregion_of(self.scope_region, scope_ub) {
            match req_mutbl {
                m_mutbl => {
                    // We do not allow non-mutable data to be loaned
                    // out as mutable under any circumstances.
                    if cmt.mutbl != m_mutbl {
                        return Err({cmt:cmt,
                                    code:err_mutbl(req_mutbl)});
                    }
                }
                m_const | m_imm => {
                    // However, mutable data can be loaned out as
                    // immutable (and any data as const).  The
                    // `check_loans` pass will then guarantee that no
                    // writes occur for the duration of the loan.
                }
            }

            self.loans.push(Loan {
                // Note: cmt.lp must be Some(_) because otherwise this
                // loan process does not apply at all.
                lp: cmt.lp.get(),
                cmt: cmt,
                mutbl: req_mutbl
            });
            return Ok(());
        } else {
            // The loan being requested lives longer than the data
            // being loaned out!
            return Err({cmt:cmt,
                        code:err_out_of_scope(scope_ub,
                                              self.scope_region)});
        }
    }

    fn loan(&self, cmt: cmt, req_mutbl: ast::mutability) -> bckres<()> {
        debug!("loan(%s, %s)",
               self.bccx.cmt_to_repr(cmt),
               self.bccx.mut_to_str(req_mutbl));
        let _i = indenter();

        // see stable() above; should only be called when `cmt` is lendable
        if cmt.lp.is_none() {
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                ~"loan() called with non-lendable value");
        }

        match cmt.cat {
          cat_binding(_) | cat_rvalue | cat_special(_) => {
            // should never be loanable
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                ~"rvalue with a non-none lp");
          }
          cat_local(local_id) | cat_arg(local_id) => {
            let local_scope_id = self.tcx().region_map.get(local_id);
            self.issue_loan(cmt, ty::re_scope(local_scope_id), req_mutbl)
          }
          cat_stack_upvar(cmt) => {
            self.loan(cmt, req_mutbl) // NDM correct?
          }
          cat_discr(base, _) => {
            self.loan(base, req_mutbl)
          }
          cat_comp(cmt_base, comp_field(_, m)) |
          cat_comp(cmt_base, comp_index(_, m)) => {
            // For most components, the type of the embedded data is
            // stable.  Therefore, the base structure need only be
            // const---unless the component must be immutable.  In
            // that case, it must also be embedded in an immutable
            // location, or else the whole structure could be
            // overwritten and the component along with it.
            self.loan_stable_comp(cmt, cmt_base, req_mutbl, m)
          }
          cat_comp(cmt_base, comp_tuple) |
          cat_comp(cmt_base, comp_anon_field) => {
            // As above.
            self.loan_stable_comp(cmt, cmt_base, req_mutbl, m_imm)
          }
          cat_comp(cmt_base, comp_variant(enum_did)) => {
            // For enums, the memory is unstable if there are multiple
            // variants, because if the enum value is overwritten then
            // the memory changes type.
            if ty::enum_is_univariant(self.bccx.tcx, enum_did) {
                self.loan_stable_comp(cmt, cmt_base, req_mutbl, m_imm)
            } else {
                self.loan_unstable_deref(cmt, cmt_base, req_mutbl)
            }
          }
          cat_deref(cmt_base, _, uniq_ptr) => {
            // For unique pointers, the memory being pointed out is
            // unstable because if the unique pointer is overwritten
            // then the memory is freed.
            self.loan_unstable_deref(cmt, cmt_base, req_mutbl)
          }
          cat_deref(_, _, unsafe_ptr) |
          cat_deref(_, _, gc_ptr) |
          cat_deref(_, _, region_ptr(_)) => {
            // Aliased data is simply not lendable.
            self.bccx.tcx.sess.span_bug(
                cmt.span,
                ~"aliased ptr with a non-none lp");
          }
        }
    }

    // A "stable component" is one where assigning the base of the
    // component cannot cause the component itself to change types.
    // Example: record fields.
    fn loan_stable_comp(&self,
                        cmt: cmt,
                        cmt_base: cmt,
                        req_mutbl: ast::mutability,
                        comp_mutbl: ast::mutability) -> bckres<()> {
        // Determine the mutability that the base component must have,
        // given the required mutability of the pointer (`req_mutbl`)
        // and the declared mutability of the component (`comp_mutbl`).
        // This is surprisingly subtle.
        //
        // Note that the *declared* mutability of the component is not
        // necessarily the same as cmt.mutbl, since a component
        // declared as immutable but embedded in a mutable context
        // becomes mutable.  It's best to think of comp_mutbl as being
        // either MUTABLE or DEFAULT, not MUTABLE or IMMUTABLE.  We
        // should really patch up the AST to reflect this distinction.
        //
        // Let's consider the cases below:
        //
        // 1. mut required, mut declared: In this case, the base
        //    component must merely be const.  The reason is that it
        //    does not matter if the base component is borrowed as
        //    mutable or immutable, as the mutability of the base
        //    component is overridden in the field declaration itself
        //    (see `compile-fail/borrowck-mut-field-imm-base.rs`)
        //
        // 2. mut required, imm declared: This would only be legal if
        //    the component is embeded in a mutable context.  However,
        //    we detect mismatches between the mutability of the value
        //    as a whole and the required mutability in `issue_loan()`
        //    above.  In any case, presuming that the component IS
        //    embedded in a mutable context, both the component and
        //    the base must be loaned as MUTABLE.  This is to ensure
        //    that there is no loan of the base as IMMUTABLE, which
        //    would imply that the component must be IMMUTABLE too
        //    (see `compile-fail/borrowck-imm-field-imm-base.rs`).
        //
        // 3. mut required, const declared: this shouldn't really be
        //    possible, since I don't think you can declare a const
        //    field, but I guess if we DID permit such a declaration
        //    it would be equivalent to the case above?
        //
        // 4. imm required, * declared: In this case, the base must be
        //    immutable.  This is true regardless of what was declared
        //    for this subcomponent, this if the base is mutable, the
        //    subcomponent must be mutable.
        //    (see `compile-fail/borrowck-imm-field-mut-base.rs`).
        //
        // 5. const required, * declared: In this case, the base need
        //    only be const, since we don't ultimately care whether
        //    the subcomponent is mutable or not.
        let base_mutbl = match (req_mutbl, comp_mutbl) {
            (m_mutbl, m_mutbl) => m_const, // (1)
            (m_mutbl, _) => m_mutbl,       // (2, 3)
            (m_imm, _) => m_imm,           // (4)
            (m_const, _) => m_const        // (5)
        };

        do self.loan(cmt_base, base_mutbl).chain |_ok| {
            // can use static for the scope because the base
            // determines the lifetime, ultimately
            self.issue_loan(cmt, ty::re_static, req_mutbl)
        }
    }

    // An "unstable deref" means a deref of a ptr/comp where, if the
    // base of the deref is assigned to, pointers into the result of the
    // deref would be invalidated. Examples: interior of variants, uniques.
    fn loan_unstable_deref(&self,
                           cmt: cmt,
                           cmt_base: cmt,
                           req_mutbl: ast::mutability) -> bckres<()> {
        // Variant components: the base must be immutable, because
        // if it is overwritten, the types of the embedded data
        // could change.
        do self.loan(cmt_base, m_imm).chain |_ok| {
            // can use static, as in loan_stable_comp()
            self.issue_loan(cmt, ty::re_static, req_mutbl)
        }
    }
}
