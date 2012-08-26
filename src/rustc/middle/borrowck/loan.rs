// ----------------------------------------------------------------------
// Loan(Ex, M, S) = Ls holds if ToAddr(Ex) will remain valid for the entirety
// of the scope S, presuming that the returned set of loans `Ls` are honored.

export public_methods;
import result::{Result, Ok, Err};

impl borrowck_ctxt {
    fn loan(cmt: cmt,
            scope_region: ty::region,
            mutbl: ast::mutability) -> bckres<@DVec<loan>> {
        let lc = loan_ctxt_(@{bccx: self,
                              scope_region: scope_region,
                              loans: @dvec()});
        match lc.loan(cmt, mutbl) {
          Ok(()) => {Ok(lc.loans)}
          Err(e) => {Err(e)}
        }
    }
}

type loan_ctxt_ = {
    bccx: borrowck_ctxt,

    // the region scope for which we must preserve the memory
    scope_region: ty::region,

    // accumulated list of loans that will be required
    loans: @DVec<loan>
};

enum loan_ctxt {
    loan_ctxt_(@loan_ctxt_)
}

impl loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn ok_with_loan_of(cmt: cmt,
                       scope_ub: ty::region,
                       mutbl: ast::mutability) -> bckres<()> {
        if self.bccx.is_subregion_of(self.scope_region, scope_ub) {
            // Note: all cmt's that we deal with will have a non-none
            // lp, because the entry point into this routine,
            // `borrowck_ctxt::loan()`, rejects any cmt with a
            // none-lp.
            (*self.loans).push({lp: option::get(cmt.lp),
                                cmt: cmt,
                                mutbl: mutbl});
            Ok(())
        } else {
            // The loan being requested lives longer than the data
            // being loaned out!
            Err({cmt:cmt, code:err_out_of_scope(scope_ub,
                                                self.scope_region)})
        }
    }

    fn loan(cmt: cmt, req_mutbl: ast::mutability) -> bckres<()> {
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
            self.ok_with_loan_of(cmt, ty::re_scope(local_scope_id), req_mutbl)
          }
          cat_stack_upvar(cmt) => {
            self.loan(cmt, req_mutbl) // NDM correct?
          }
          cat_discr(base, _) => {
            self.loan(base, req_mutbl)
          }
          cat_comp(cmt_base, comp_field(*)) |
          cat_comp(cmt_base, comp_index(*)) |
          cat_comp(cmt_base, comp_tuple) => {
            // For most components, the type of the embedded data is
            // stable.  Therefore, the base structure need only be
            // const---unless the component must be immutable.  In
            // that case, it must also be embedded in an immutable
            // location, or else the whole structure could be
            // overwritten and the component along with it.
            self.loan_stable_comp(cmt, cmt_base, req_mutbl)
          }
          cat_comp(cmt_base, comp_variant(enum_did)) => {
            // For enums, the memory is unstable if there are multiple
            // variants, because if the enum value is overwritten then
            // the memory changes type.
            if ty::enum_is_univariant(self.bccx.tcx, enum_did) {
                self.loan_stable_comp(cmt, cmt_base, req_mutbl)
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
    fn loan_stable_comp(cmt: cmt,
                        cmt_base: cmt,
                        req_mutbl: ast::mutability) -> bckres<()> {
        let base_mutbl = match req_mutbl {
          m_imm => m_imm,
          m_const | m_mutbl => m_const
        };

        do self.loan(cmt_base, base_mutbl).chain |_ok| {
            // can use static for the scope because the base
            // determines the lifetime, ultimately
            self.ok_with_loan_of(cmt, ty::re_static, req_mutbl)
        }
    }

    // An "unstable deref" means a deref of a ptr/comp where, if the
    // base of the deref is assigned to, pointers into the result of the
    // deref would be invalidated. Examples: interior of variants, uniques.
    fn loan_unstable_deref(cmt: cmt,
                           cmt_base: cmt,
                           req_mutbl: ast::mutability) -> bckres<()> {
        // Variant components: the base must be immutable, because
        // if it is overwritten, the types of the embedded data
        // could change.
        do self.loan(cmt_base, m_imm).chain |_ok| {
            // can use static, as in loan_stable_comp()
            self.ok_with_loan_of(cmt, ty::re_static, req_mutbl)
        }
    }
}
