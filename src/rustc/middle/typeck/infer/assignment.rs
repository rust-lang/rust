// ______________________________________________________________________
// Type assignment
//
// True if rvalues of type `a` can be assigned to lvalues of type `b`.
// This may cause borrowing to the region scope enclosing `a_node_id`.
//
// The strategy here is somewhat non-obvious.  The problem is
// that the constraint we wish to contend with is not a subtyping
// constraint.  Currently, for variables, we only track what it
// must be a subtype of, not what types it must be assignable to
// (or from).  Possibly, we should track that, but I leave that
// refactoring for another day.
//
// Instead, we look at each variable involved and try to extract
// *some* sort of bound.  Typically, the type a is the argument
// supplied to a call; it typically has a *lower bound* (which
// comes from having been assigned a value).  What we'd actually
// *like* here is an upper-bound, but we generally don't have
// one.  The type b is the expected type and it typically has a
// lower-bound too, which is good.
//
// The way we deal with the fact that we often don't have the
// bounds we need is to be a bit careful.  We try to get *some*
// bound from each side, preferring the upper from a and the
// lower from b.  If we fail to get a bound from both sides, then
// we just fall back to requiring that a <: b.
//
// Assuming we have a bound from both sides, we will then examine
// these bounds and see if they have the form (@M_a T_a, &rb.M_b T_b)
// (resp. ~M_a T_a, ~[M_a T_a], etc).  If they do not, we fall back to
// subtyping.
//
// If they *do*, then we know that the two types could never be
// subtypes of one another.  We will then construct a type @const T_b
// and ensure that type a is a subtype of that.  This allows for the
// possibility of assigning from a type like (say) @~[mut T1] to a type
// &~[T2] where T1 <: T2.  This might seem surprising, since the `@`
// points at mutable memory but the `&` points at immutable memory.
// This would in fact be unsound, except for the borrowck, which comes
// later and guarantees that such mutability conversions are safe.
// See borrowck for more details.  Next we require that the region for
// the enclosing scope be a superregion of the region r.
//
// You might wonder why we don't make the type &e.const T_a where e is
// the enclosing region and check that &e.const T_a <: B.  The reason
// is that the type of A is (generally) just a *lower-bound*, so this
// would be imposing that lower-bound also as the upper-bound on type
// A.  But this upper-bound might be stricter than what is truly
// needed.

import to_str::to_str;

impl infer_ctxt {
    fn assign_tys(anmnt: &assignment, a: ty::t, b: ty::t) -> ures {

        fn select(fst: option<ty::t>, snd: option<ty::t>) -> option<ty::t> {
            match fst {
              some(t) => some(t),
              none => match snd {
                some(t) => some(t),
                none => none
              }
            }
        }

        debug!{"assign_tys(anmnt=%?, %s -> %s)",
               anmnt, a.to_str(self), b.to_str(self)};
        let _r = indenter();

        match (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_bot, _) => {
            uok()
          }

          (ty::ty_var(a_id), ty::ty_var(b_id)) => {
            let nde_a = self.get(&self.ty_var_bindings, a_id);
            let nde_b = self.get(&self.ty_var_bindings, b_id);
            let a_bounds = nde_a.possible_types;
            let b_bounds = nde_b.possible_types;

            let a_bnd = select(a_bounds.ub, a_bounds.lb);
            let b_bnd = select(b_bounds.lb, b_bounds.ub);
            self.assign_tys_or_sub(anmnt, a, b, a_bnd, b_bnd)
          }

          (ty::ty_var(a_id), _) => {
            let nde_a = self.get(&self.ty_var_bindings, a_id);
            let a_bounds = nde_a.possible_types;

            let a_bnd = select(a_bounds.ub, a_bounds.lb);
            self.assign_tys_or_sub(anmnt, a, b, a_bnd, some(b))
          }

          (_, ty::ty_var(b_id)) => {
            let nde_b = self.get(&self.ty_var_bindings, b_id);
            let b_bounds = nde_b.possible_types;

            let b_bnd = select(b_bounds.lb, b_bounds.ub);
            self.assign_tys_or_sub(anmnt, a, b, some(a), b_bnd)
          }

          (_, _) => {
            self.assign_tys_or_sub(anmnt, a, b, some(a), some(b))
          }
        }
    }

    fn assign_tys_or_sub(
        anmnt: &assignment,
        a: ty::t, b: ty::t,
        +a_bnd: option<ty::t>, +b_bnd: option<ty::t>) -> ures {

        debug!{"assign_tys_or_sub(anmnt=%?, %s -> %s, %s -> %s)",
               anmnt, a.to_str(self), b.to_str(self),
               a_bnd.to_str(self), b_bnd.to_str(self)};
        let _r = indenter();

        fn is_borrowable(v: ty::vstore) -> bool {
            match v {
              ty::vstore_fixed(_) | ty::vstore_uniq | ty::vstore_box => true,
              ty::vstore_slice(_) => false
            }
        }

        match (a_bnd, b_bnd) {
          (some(a_bnd), some(b_bnd)) => {
            match (ty::get(a_bnd).struct, ty::get(b_bnd).struct) {
              (ty::ty_box(mt_a), ty::ty_rptr(r_b, mt_b)) => {
                let nr_b = ty::mk_box(self.tcx, {ty: mt_b.ty,
                                                 mutbl: m_const});
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }
              (ty::ty_uniq(mt_a), ty::ty_rptr(r_b, mt_b)) => {
                let nr_b = ty::mk_uniq(self.tcx, {ty: mt_b.ty,
                                                  mutbl: m_const});
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }
              (ty::ty_estr(vs_a),
               ty::ty_estr(ty::vstore_slice(r_b)))
              if is_borrowable(vs_a) => {
                let nr_b = ty::mk_estr(self.tcx, vs_a);
                self.crosspollinate(anmnt, a, nr_b, m_imm, r_b)
              }

              (ty::ty_evec(mt_a, vs_a),
               ty::ty_evec(mt_b, ty::vstore_slice(r_b)))
              if is_borrowable(vs_a) => {
                let nr_b = ty::mk_evec(self.tcx, {ty: mt_b.ty,
                                                  mutbl: m_const}, vs_a);
                self.crosspollinate(anmnt, a, nr_b, mt_b.mutbl, r_b)
              }

              _ => {
                mk_sub(self, false, anmnt.span).tys(a, b).to_ures()
              }
            }
          }
          _ => {
            mk_sub(self, false, anmnt.span).tys(a, b).to_ures()
          }
        }
    }

    fn crosspollinate(anmnt: &assignment,
                      a: ty::t,
                      nr_b: ty::t,
                      m: ast::mutability,
                      r_b: ty::region) -> ures {

        debug!{"crosspollinate(anmnt=%?, a=%s, nr_b=%s, r_b=%s)",
               anmnt, a.to_str(self), nr_b.to_str(self),
               r_b.to_str(self)};

        do indent {
            let sub = mk_sub(self, false, anmnt.span);
            do sub.tys(a, nr_b).chain |_t| {
                // Create a fresh region variable `r_a` with the given
                // borrow bounds:
                let r_a = self.next_region_var(anmnt.span,
                                               anmnt.borrow_lb);

                debug!{"anmnt=%?", anmnt};
                do sub.contraregions(r_a, r_b).chain |_r| {
                    // if successful, add an entry indicating that
                    // borrowing occurred
                    debug!{"borrowing expression #%?, scope=%?, m=%?",
                           anmnt, r_a, m};
                    self.borrowings.push({expr_id: anmnt.expr_id,
                                          span: anmnt.span,
                                          scope: r_a,
                                          mutbl: m});
                    uok()
                }
            }
        }
    }
}

