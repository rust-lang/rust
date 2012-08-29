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
import combine::combine_fields;

fn to_ares(+c: cres<ty::t>) -> ares {
    match c {
        Ok(_) => Ok(None),
        Err(e) => Err(e)
    }
}

// Note: Assign is not actually a combiner, in that it does not
// conform to the same interface, though it performs a similar
// function.
enum Assign = combine_fields;

impl Assign {
    fn tys(a: ty::t, b: ty::t) -> ares {
        debug!("Assign.tys(%s -> %s)",
               a.to_str(self.infcx),
               b.to_str(self.infcx));
        let _r = indenter();

        match (ty::get(a).struct, ty::get(b).struct) {
            (ty::ty_bot, _) => {
                Ok(None)
            }

            (ty::ty_var(a_id), ty::ty_var(b_id)) => {
                let nde_a = self.infcx.get(&self.infcx.ty_var_bindings, a_id);
                let nde_b = self.infcx.get(&self.infcx.ty_var_bindings, b_id);
                let a_bounds = nde_a.possible_types;
                let b_bounds = nde_b.possible_types;

                let a_bnd = option::or(a_bounds.ub, a_bounds.lb);
                let b_bnd = option::or(b_bounds.lb, b_bounds.ub);
                self.assign_tys_or_sub(a, b, a_bnd, b_bnd)
            }

            (ty::ty_var(a_id), _) => {
                let nde_a = self.infcx.get(&self.infcx.ty_var_bindings, a_id);
                let a_bounds = nde_a.possible_types;

                let a_bnd = option::or(a_bounds.ub, a_bounds.lb);
                self.assign_tys_or_sub(a, b, a_bnd, Some(b))
            }

            (_, ty::ty_var(b_id)) => {
                let nde_b = self.infcx.get(&self.infcx.ty_var_bindings, b_id);
                let b_bounds = nde_b.possible_types;

                let b_bnd = option::or(b_bounds.lb, b_bounds.ub);
                self.assign_tys_or_sub(a, b, Some(a), b_bnd)
            }

            (_, _) => {
                self.assign_tys_or_sub(a, b, Some(a), Some(b))
            }
        }
    }
}

priv impl Assign {
    fn assign_tys_or_sub(
        a: ty::t, b: ty::t,
        +a_bnd: Option<ty::t>, +b_bnd: Option<ty::t>) -> ares {

        debug!("Assign.assign_tys_or_sub(%s -> %s, %s -> %s)",
               a.to_str(self.infcx), b.to_str(self.infcx),
               a_bnd.to_str(self.infcx), b_bnd.to_str(self.infcx));
        let _r = indenter();

        fn is_borrowable(v: ty::vstore) -> bool {
            match v {
              ty::vstore_fixed(_) | ty::vstore_uniq | ty::vstore_box => true,
              ty::vstore_slice(_) => false
            }
        }

        match (a_bnd, b_bnd) {
            (Some(a_bnd), Some(b_bnd)) => {
                // check for a case where a non-region pointer (@, ~) is
                // being assigned to a region pointer:
                match (ty::get(a_bnd).struct, ty::get(b_bnd).struct) {
                    (ty::ty_box(_), ty::ty_rptr(r_b, mt_b)) => {
                        let nr_b = ty::mk_box(self.infcx.tcx,
                                              {ty: mt_b.ty, mutbl: m_const});
                        self.try_assign(a, nr_b, mt_b.mutbl, r_b)
                    }
                    (ty::ty_uniq(_), ty::ty_rptr(r_b, mt_b)) => {
                        let nr_b = ty::mk_uniq(self.infcx.tcx,
                                               {ty: mt_b.ty, mutbl: m_const});
                        self.try_assign(a, nr_b, mt_b.mutbl, r_b)
                    }
                    (ty::ty_estr(vs_a),
                     ty::ty_estr(ty::vstore_slice(r_b)))
                    if is_borrowable(vs_a) => {
                        let nr_b = ty::mk_estr(self.infcx.tcx, vs_a);
                        self.try_assign(a, nr_b, m_imm, r_b)
                    }

                    (ty::ty_evec(_, vs_a),
                     ty::ty_evec(mt_b, ty::vstore_slice(r_b)))
                    if is_borrowable(vs_a) => {
                        let nr_b = ty::mk_evec(self.infcx.tcx,
                                               {ty: mt_b.ty, mutbl: m_const},
                                               vs_a);
                        self.try_assign(a, nr_b, mt_b.mutbl, r_b)
                    }

                    _ => {
                        // otherwise, assignment follows normal subtype rules:
                        to_ares(Sub(*self).tys(a, b))
                    }
                }
            }
            _ => {
                // if insufficient bounds were available, just follow
                // normal subtype rules:
                to_ares(Sub(*self).tys(a, b))
            }
        }
    }

    /// Given an assignment from a type like `@a` to `&r_b/m nr_b`,
    /// this function checks that `a <: nr_b`.  In that case, the
    /// assignment is permitted, so it constructs a fresh region
    /// variable `r_a >= r_b` and returns a corresponding assignment
    /// record.  See the discussion at the top of this file for more
    /// details.
    fn try_assign(a: ty::t,
                  nr_b: ty::t,
                  m: ast::mutability,
                  r_b: ty::region) -> ares {

        debug!("try_assign(a=%s, nr_b=%s, m=%?, r_b=%s)",
               a.to_str(self.infcx),
               nr_b.to_str(self.infcx),
               m,
               r_b.to_str(self.infcx));

        do indent {
            let sub = Sub(*self);
            do sub.tys(a, nr_b).chain |_t| {
                let r_a = self.infcx.next_region_var_nb(self.span);
                do sub.contraregions(r_a, r_b).chain |_r| {
                    Ok(Some({region: r_a, mutbl: m}))
                }
            }
        }
    }
}

