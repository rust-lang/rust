import combine::*;
import unify::*;
import to_str::to_str;

// ______________________________________________________________________
// Lattice operations on variables
//
// This is common code used by both LUB and GLB to compute the LUB/GLB
// for pairs of variables or for variables and values.

trait lattice_ops {
    fn bnd(b: bounds<ty::t>) -> option<ty::t>;
    fn with_bnd(b: bounds<ty::t>, t: ty::t) -> bounds<ty::t>;
    fn ty_bot(t: ty::t) -> cres<ty::t>;
}

impl Lub: lattice_ops {
    fn bnd(b: bounds<ty::t>) -> option<ty::t> { b.ub }
    fn with_bnd(b: bounds<ty::t>, t: ty::t) -> bounds<ty::t> {
        {ub: some(t) with b}
    }
    fn ty_bot(t: ty::t) -> cres<ty::t> {
        ok(t)
    }
}

impl Glb: lattice_ops {
    fn bnd(b: bounds<ty::t>) -> option<ty::t> { b.lb }
    fn with_bnd(b: bounds<ty::t>, t: ty::t) -> bounds<ty::t> {
        {lb: some(t) with b}
    }
    fn ty_bot(_t: ty::t) -> cres<ty::t> {
        ok(ty::mk_bot(self.infcx.tcx))
    }
}

fn lattice_tys<L:lattice_ops combine>(
    self: &L, a: ty::t, b: ty::t) -> cres<ty::t> {

    debug!{"%s.lattice_tys(%s, %s)", self.tag(),
           a.to_str(self.infcx()),
           b.to_str(self.infcx())};
    if a == b { return ok(a); }
    do indent {
        match (ty::get(a).struct, ty::get(b).struct) {
          (ty::ty_bot, _) => self.ty_bot(b),
          (_, ty::ty_bot) => self.ty_bot(a),

          (ty::ty_var(a_id), ty::ty_var(b_id)) => {
            lattice_vars(self, a, a_id, b_id,
                         |x, y| self.tys(x, y) )
          }

          (ty::ty_var(a_id), _) => {
            lattice_var_and_t(self, a_id, b,
                              |x, y| self.tys(x, y) )
          }

          (_, ty::ty_var(b_id)) => {
            lattice_var_and_t(self, b_id, a,
                              |x, y| self.tys(x, y) )
          }
          _ => {
            super_tys(self, a, b)
          }
        }
    }
}

fn lattice_vars<L:lattice_ops combine>(
    self: &L, +a_t: ty::t, +a_vid: ty::tv_vid, +b_vid: ty::tv_vid,
    c_ts: fn(ty::t, ty::t) -> cres<ty::t>) -> cres<ty::t> {

    // The comments in this function are written for LUB and types,
    // but they apply equally well to GLB and regions if you inverse
    // upper/lower/sub/super/etc.

    // Need to find a type that is a supertype of both a and b:
    let vb = &self.infcx().ty_var_bindings;
    let nde_a = self.infcx().get(vb, a_vid);
    let nde_b = self.infcx().get(vb, b_vid);
    let a_vid = nde_a.root;
    let b_vid = nde_b.root;
    let a_bounds = nde_a.possible_types;
    let b_bounds = nde_b.possible_types;

    debug!{"%s.lattice_vars(%s=%s <: %s=%s)",
           self.tag(),
           a_vid.to_str(), a_bounds.to_str(self.infcx()),
           b_vid.to_str(), b_bounds.to_str(self.infcx())};

    if a_vid == b_vid {
        return ok(a_t);
    }

    // If both A and B have an UB type, then we can just compute the
    // LUB of those types:
    let a_bnd = self.bnd(a_bounds), b_bnd = self.bnd(b_bounds);
    match (a_bnd, b_bnd) {
      (some(a_ty), some(b_ty)) => {
        match self.infcx().try(|| c_ts(a_ty, b_ty) ) {
            ok(t) => return ok(t),
            err(_) => { /*fallthrough */ }
        }
      }
      _ => {/*fallthrough*/}
    }

    // Otherwise, we need to merge A and B into one variable.  We can
    // then use either variable as an upper bound:
    var_sub_var(self, a_vid, b_vid).then(|| ok(a_t) )
}

fn lattice_var_and_t<L:lattice_ops combine>(
    self: &L, a_id: ty::tv_vid, b: ty::t,
    c_ts: fn(ty::t, ty::t) -> cres<ty::t>) -> cres<ty::t> {

    let vb = &self.infcx().ty_var_bindings;
    let nde_a = self.infcx().get(vb, a_id);
    let a_id = nde_a.root;
    let a_bounds = nde_a.possible_types;

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    debug!{"%s.lattice_var_and_t(%s=%s <: %s)",
           self.tag(),
           a_id.to_str(), a_bounds.to_str(self.infcx()),
           b.to_str(self.infcx())};

    match self.bnd(a_bounds) {
      some(a_bnd) => {
        // If a has an upper bound, return the LUB(a.ub, b)
        debug!{"bnd=some(%s)", a_bnd.to_str(self.infcx())};
        return c_ts(a_bnd, b);
      }
      none => {
        // If a does not have an upper bound, make b the upper bound of a
        // and then return b.
        debug!{"bnd=none"};
        let a_bounds = self.with_bnd(a_bounds, b);
        do bnds(self, a_bounds.lb, a_bounds.ub).then {
            self.infcx().set(vb, a_id, root(a_bounds, nde_a.rank));
            ok(b)
        }
      }
    }
}
