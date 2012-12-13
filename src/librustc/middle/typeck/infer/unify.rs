// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::typeck::infer::combine::combine;
use middle::typeck::infer::floating::*;
use middle::typeck::infer::integral::*;
use middle::typeck::infer::to_str::ToStr;

use std::smallintmap::SmallIntMap;

enum var_value<V:Copy, T:Copy> {
    redirect(V),
    root(T, uint),
}

struct vals_and_bindings<V:Copy, T:Copy> {
    vals: SmallIntMap<var_value<V, T>>,
    mut bindings: ~[(V, var_value<V, T>)],
}

struct node<V:Copy, T:Copy> {
    root: V,
    possible_types: T,
    rank: uint,
}

impl infer_ctxt {
    fn get<V:Copy vid Eq, T:Copy>(
        vb: &vals_and_bindings<V, T>, vid: V) -> node<V, T> {

        let vid_u = vid.to_uint();
        match vb.vals.find(vid_u) {
          None => {
            self.tcx.sess.bug(fmt!("failed lookup of vid `%u`", vid_u));
          }
          Some(ref var_val) => {
            match (*var_val) {
              redirect(ref vid) => {
                let node = self.get(vb, (*vid));
                if node.root.ne(vid) {
                    // Path compression
                    vb.vals.insert((*vid).to_uint(), redirect(node.root));
                }
                node
              }
              root(ref pt, rk) => {
                node {root: vid, possible_types: (*pt), rank: rk}
              }
            }
          }
        }
    }

    fn set<V:Copy vid, T:Copy ToStr>(
        vb: &vals_and_bindings<V, T>, vid: V,
        +new_v: var_value<V, T>) {

        let old_v = vb.vals.get(vid.to_uint());
        vb.bindings.push((vid, old_v));
        vb.vals.insert(vid.to_uint(), new_v);

        debug!("Updating variable %s from %s to %s",
               vid.to_str(), old_v.to_str(self), new_v.to_str(self));
    }
}

// Combines the two bounds into a more general bound.
fn merge_bnd<C: combine>(
    self: &C, a: bound<ty::t>, b: bound<ty::t>,
    merge_op: fn(ty::t,ty::t) -> cres<ty::t>) -> cres<bound<ty::t>> {

    debug!("merge_bnd(%s,%s)",
           a.to_str(self.infcx()),
           b.to_str(self.infcx()));
    let _r = indenter();

    match (a, b) {
      (None, None) => Ok(None),
      (Some(_), None) => Ok(a),
      (None, Some(_)) => Ok(b),
      (Some(v_a), Some(v_b)) => {
        do merge_op(v_a, v_b).chain |v| {
            Ok(Some(v))
        }
      }
    }
}

fn merge_bnds<C: combine>(
    self: &C, a: bounds<ty::t>, b: bounds<ty::t>,
    lub: fn(ty::t,ty::t) -> cres<ty::t>,
    glb: fn(ty::t,ty::t) -> cres<ty::t>) -> cres<bounds<ty::t>> {

    let _r = indenter();
    do merge_bnd(self, a.ub, b.ub, glb).chain |ub| {
        debug!("glb of ubs %s and %s is %s",
               a.ub.to_str(self.infcx()),
               b.ub.to_str(self.infcx()),
               ub.to_str(self.infcx()));
        do merge_bnd(self, a.lb, b.lb, lub).chain |lb| {
            debug!("lub of lbs %s and %s is %s",
                   a.lb.to_str(self.infcx()),
                   b.lb.to_str(self.infcx()),
                   lb.to_str(self.infcx()));
            Ok({lb: lb, ub: ub})
        }
    }
}

// Updates the bounds for the variable `v_id` to be the intersection
// of `a` and `b`.  That is, the new bounds for `v_id` will be
// a bounds c such that:
//    c.ub <: a.ub
//    c.ub <: b.ub
//    a.lb <: c.lb
//    b.lb <: c.lb
// If this cannot be achieved, the result is failure.

fn set_var_to_merged_bounds<C: combine>(
    self: &C,
    v_id: ty::TyVid,
    a: bounds<ty::t>,
    b: bounds<ty::t>,
    rank: uint) -> ures {

    let vb = &self.infcx().ty_var_bindings;

    // Think of the two diamonds, we want to find the
    // intersection.  There are basically four possibilities (you
    // can swap A/B in these pictures):
    //
    //       A         A
    //      / \       / \
    //     / B \     / B \
    //    / / \ \   / / \ \
    //   * *   * * * /   * *
    //    \ \ / /   \   / /
    //     \ B /   / \ / /
    //      \ /   *   \ /
    //       A     \ / A
    //              B

    debug!("merge(%s,%s,%s)",
           v_id.to_str(),
           a.to_str(self.infcx()),
           b.to_str(self.infcx()));

    // First, relate the lower/upper bounds of A and B.
    // Note that these relations *must* hold for us to
    // to be able to merge A and B at all, and relating
    // them explicitly gives the type inferencer more
    // information and helps to produce tighter bounds
    // when necessary.
    do indent {
        do bnds(self, a.lb, b.ub).then {
            do bnds(self, b.lb, a.ub).then {
                do merge_bnd(self, a.ub, b.ub,
                             |x, y| self.glb().tys(x, y)).chain |ub| {
                    do merge_bnd(self, a.lb, b.lb,
                                 |x, y| self.lub().tys(x, y)).chain |lb| {
                        let bounds = {lb: lb, ub: ub};
                        debug!("merge(%s): bounds=%s",
                               v_id.to_str(),
                               bounds.to_str(self.infcx()));

                        // the new bounds must themselves
                        // be relatable:
                        do bnds(self, bounds.lb, bounds.ub).then {
                            self.infcx().set(vb, v_id, root(bounds, rank));
                            uok()
                        }
                    }
                }
            }
        }
    }
}

/// Ensure that variable A is a subtype of variable B.  This is a
/// subtle and tricky process, as described in detail at the top
/// of infer.rs
fn var_sub_var<C: combine>(self: &C,
                           a_id: ty::TyVid,
                           b_id: ty::TyVid) -> ures {
    let vb = &self.infcx().ty_var_bindings;

    // Need to make sub_id a subtype of sup_id.
    let nde_a = self.infcx().get(vb, a_id);
    let nde_b = self.infcx().get(vb, b_id);
    let a_id = nde_a.root;
    let b_id = nde_b.root;
    let a_bounds = nde_a.possible_types;
    let b_bounds = nde_b.possible_types;

    debug!("vars(%s=%s <: %s=%s)",
           a_id.to_str(), a_bounds.to_str(self.infcx()),
           b_id.to_str(), b_bounds.to_str(self.infcx()));

    if a_id == b_id { return uok(); }

    // If both A's UB and B's LB have already been bound to types,
    // see if we can make those types subtypes.
    match (a_bounds.ub, b_bounds.lb) {
      (Some(a_ub), Some(b_lb)) => {
        let r = self.infcx().try(|| self.sub().tys(a_ub, b_lb));
        match r {
          Ok(_ty) => return result::Ok(()),
          Err(_) => { /*fallthrough */ }
        }
      }
      _ => { /*fallthrough*/ }
    }

    // Otherwise, we need to merge A and B so as to guarantee that
    // A remains a subtype of B.  Actually, there are other options,
    // but that's the route we choose to take.

    // Rank optimization

    // Make the node with greater rank the parent of the node with
    // smaller rank.
    if nde_a.rank > nde_b.rank {
        debug!("vars(): a has smaller rank");
        // a has greater rank, so a should become b's parent,
        // i.e., b should redirect to a.
        self.infcx().set(vb, b_id, redirect(a_id));
        set_var_to_merged_bounds(
            self, a_id, a_bounds, b_bounds, nde_a.rank)
    } else if nde_a.rank < nde_b.rank {
        debug!("vars(): b has smaller rank");
        // b has greater rank, so a should redirect to b.
        self.infcx().set(vb, a_id, redirect(b_id));
        set_var_to_merged_bounds(
            self, b_id, a_bounds, b_bounds, nde_b.rank)
    } else {
        debug!("vars(): a and b have equal rank");
        assert nde_a.rank == nde_b.rank;
        // If equal, just redirect one to the other and increment
        // the other's rank.  We choose arbitrarily to redirect b
        // to a and increment a's rank.
        self.infcx().set(vb, b_id, redirect(a_id));
        set_var_to_merged_bounds(
            self, a_id, a_bounds, b_bounds, nde_a.rank + 1u
        )
    }
}

/// make variable a subtype of T
fn var_sub_t<C: combine>(self: &C, a_id: ty::TyVid, b: ty::t) -> ures {

    let vb = &self.infcx().ty_var_bindings;
    let nde_a = self.infcx().get(vb, a_id);
    let a_id = nde_a.root;
    let a_bounds = nde_a.possible_types;

    debug!("var_sub_t(%s=%s <: %s)",
           a_id.to_str(),
           a_bounds.to_str(self.infcx()),
           b.to_str(self.infcx()));
    let b_bounds = {lb: None, ub: Some(b)};
    set_var_to_merged_bounds(self, a_id, a_bounds, b_bounds, nde_a.rank)
}

/// make T a subtype of variable
fn t_sub_var<C: combine>(self: &C, a: ty::t, b_id: ty::TyVid) -> ures {

    let vb = &self.infcx().ty_var_bindings;
    let a_bounds = {lb: Some(a), ub: None};
    let nde_b = self.infcx().get(vb, b_id);
    let b_id = nde_b.root;
    let b_bounds = nde_b.possible_types;

    debug!("t_sub_var(%s <: %s=%s)",
           a.to_str(self.infcx()),
           b_id.to_str(),
           b_bounds.to_str(self.infcx()));
    set_var_to_merged_bounds(self, b_id, a_bounds, b_bounds, nde_b.rank)
}

fn bnds<C: combine>(
    self: &C, a: bound<ty::t>, b: bound<ty::t>) -> ures {

    debug!("bnds(%s <: %s)", a.to_str(self.infcx()), b.to_str(self.infcx()));
    do indent {
        match (a, b) {
          (None, None) |
          (Some(_), None) |
          (None, Some(_)) => {
            uok()
          }
          (Some(t_a), Some(t_b)) => {
            self.sub().tys(t_a, t_b).to_ures()
          }
        }
    }
}

// ______________________________________________________________________
// Integral variables

impl infer_ctxt {
    fn optimize_ranks<V:Copy vid Eq,T:Copy ToStr>(vb: &vals_and_bindings<V,T>,
                                                  nde_a: node<V,T>,
                                                  nde_b: node<V,T>,
                                                  a_id: V,
                                                  b_id: V,
                                                  intersection: T) {
        if nde_a.rank > nde_b.rank {
            debug!("int_vars(): a has smaller rank");
            // a has greater rank, so a should become b's parent,
            // i.e., b should redirect to a.
            self.set(vb, a_id, root(intersection, nde_a.rank));
            self.set(vb, b_id, redirect(a_id));
        } else if nde_a.rank < nde_b.rank {
            debug!("int_vars(): b has smaller rank");
            // b has greater rank, so a should redirect to b.
            self.set(vb, b_id, root(intersection, nde_b.rank));
            self.set(vb, a_id, redirect(b_id));
        } else {
            debug!("int_vars(): a and b have equal rank");
            assert nde_a.rank == nde_b.rank;
            // If equal, just redirect one to the other and increment
            // the other's rank.  We choose arbitrarily to redirect b
            // to a and increment a's rank.
            self.set(vb, a_id, root(intersection, nde_a.rank + 1u));
            self.set(vb, b_id, redirect(a_id));
        };
    }

    fn int_vars(a_id: ty::IntVid, b_id: ty::IntVid) -> ures {
        let vb = &self.int_var_bindings;

        let nde_a = self.get(vb, a_id);
        let nde_b = self.get(vb, b_id);
        let a_id = nde_a.root;
        let b_id = nde_b.root;
        let a_pt = nde_a.possible_types;
        let b_pt = nde_b.possible_types;

        // If we're already dealing with the same two variables,
        // there's nothing to do.
        if a_id == b_id { return uok(); }

        // Otherwise, take the intersection of the two sets of
        // possible types.
        let intersection = integral::intersection(a_pt, b_pt);
        if *intersection == INT_TY_SET_EMPTY {
            return Err(ty::terr_no_integral_type);
        }

        // Rank optimization
        self.optimize_ranks(vb, nde_a, nde_b, a_id, b_id, intersection);

        uok()
    }

    fn int_var_sub_t(a_id: ty::IntVid, b: ty::t) -> ures {
        assert ty::type_is_integral(b);

        let vb = &self.int_var_bindings;
        let nde_a = self.get(vb, a_id);
        let a_id = nde_a.root;
        let a_pt = nde_a.possible_types;

        let intersection =
            integral::intersection(a_pt,
                         convert_integral_ty_to_int_ty_set(self.tcx, b));
        if *intersection == INT_TY_SET_EMPTY {
            return Err(ty::terr_no_integral_type);
        }
        self.set(vb, a_id, root(intersection, nde_a.rank));
        uok()
    }

    fn t_sub_int_var(a: ty::t, b_id: ty::IntVid) -> ures {
        assert ty::type_is_integral(a);
        let vb = &self.int_var_bindings;

        let nde_b = self.get(vb, b_id);
        let b_id = nde_b.root;
        let b_pt = nde_b.possible_types;

        let intersection =
            integral::intersection(b_pt,
                         convert_integral_ty_to_int_ty_set(self.tcx, a));
        if *intersection == INT_TY_SET_EMPTY {
            return Err(ty::terr_no_integral_type);
        }
        self.set(vb, b_id, root(intersection, nde_b.rank));
        uok()
    }


}

// ______________________________________________________________________
// Floating point variables

impl infer_ctxt {
    fn float_vars(a_id: ty::FloatVid, b_id: ty::FloatVid) -> ures {
        let vb = &self.float_var_bindings;

        let nde_a = self.get(vb, a_id);
        let nde_b = self.get(vb, b_id);
        let a_id = nde_a.root;
        let b_id = nde_b.root;
        let a_pt = nde_a.possible_types;
        let b_pt = nde_b.possible_types;

        // If we're already dealing with the same two variables,
        // there's nothing to do.
        if a_id == b_id { return uok(); }

        // Otherwise, take the intersection of the two sets of
        // possible types.
        let intersection = floating::intersection(a_pt, b_pt);
        if *intersection == FLOAT_TY_SET_EMPTY {
            return Err(ty::terr_no_floating_point_type);
        }

        // Rank optimization
        self.optimize_ranks(vb, nde_a, nde_b, a_id, b_id, intersection);

        uok()
    }

    fn float_var_sub_t(a_id: ty::FloatVid, b: ty::t) -> ures {
        assert ty::type_is_fp(b);

        let vb = &self.float_var_bindings;
        let nde_a = self.get(vb, a_id);
        let a_id = nde_a.root;
        let a_pt = nde_a.possible_types;

        let intersection =
            floating::intersection(
                a_pt,
                convert_floating_point_ty_to_float_ty_set(self.tcx, b));
        if *intersection == FLOAT_TY_SET_EMPTY {
            return Err(ty::terr_no_floating_point_type);
        }
        self.set(vb, a_id, root(intersection, nde_a.rank));
        uok()
    }

    fn t_sub_float_var(a: ty::t, b_id: ty::FloatVid) -> ures {
        assert ty::type_is_fp(a);
        let vb = &self.float_var_bindings;

        let nde_b = self.get(vb, b_id);
        let b_id = nde_b.root;
        let b_pt = nde_b.possible_types;

        let intersection =
            floating::intersection(
                b_pt,
                convert_floating_point_ty_to_float_ty_set(self.tcx, a));
        if *intersection == FLOAT_TY_SET_EMPTY {
            return Err(ty::terr_no_floating_point_type);
        }
        self.set(vb, b_id, root(intersection, nde_b.rank));
        uok()
    }
}

