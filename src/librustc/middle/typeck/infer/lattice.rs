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
 *
 * # Lattice Variables
 *
 * This file contains generic code for operating on inference variables
 * that are characterized by an upper- and lower-bound.  The logic and
 * reasoning is explained in detail in the large comment in `infer.rs`.
 *
 * The code in here is defined quite generically so that it can be
 * applied both to type variables, which represent types being inferred,
 * and fn variables, which represent function types being inferred.
 * It may eventually be applied to ther types as well, who knows.
 * In some cases, the functions are also generic with respect to the
 * operation on the lattice (GLB vs LUB).
 *
 * Although all the functions are generic, we generally write the
 * comments in a way that is specific to type variables and the LUB
 * operation.  It's just easier that way.
 *
 * In general all of the functions are defined parametrically
 * over a `LatticeValue`, which is a value defined with respect to
 * a lattice.
 */


use middle::ty::{RegionVid, TyVar, Vid};
use middle::ty;
use middle::typeck::infer::{then, ToUres};
use middle::typeck::infer::*;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::unify::*;
use middle::typeck::infer::sub::Sub;
use middle::typeck::infer::to_str::InferStr;
use util::common::indenter;

use collections::HashMap;
use std::vec::Vec;

pub trait LatticeValue {
    fn sub(cf: &CombineFields, a: &Self, b: &Self) -> ures;
    fn lub(cf: &CombineFields, a: &Self, b: &Self) -> cres<Self>;
    fn glb(cf: &CombineFields, a: &Self, b: &Self) -> cres<Self>;
}

pub type LatticeOp<'a, T> =
    'a |cf: &CombineFields, a: &T, b: &T| -> cres<T>;

impl LatticeValue for ty::t {
    fn sub(cf: &CombineFields, a: &ty::t, b: &ty::t) -> ures {
        Sub(*cf).tys(*a, *b).to_ures()
    }

    fn lub(cf: &CombineFields, a: &ty::t, b: &ty::t) -> cres<ty::t> {
        Lub(*cf).tys(*a, *b)
    }

    fn glb(cf: &CombineFields, a: &ty::t, b: &ty::t) -> cres<ty::t> {
        Glb(*cf).tys(*a, *b)
    }
}

pub trait CombineFieldsLatticeMethods {
    fn var_sub_var<T:Clone + InferStr + LatticeValue,
                   V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(&self,
                                                                     a_id: V,
                                                                     b_id: V)
                                                                     -> ures;
    /// make variable a subtype of T
    fn var_sub_t<T:Clone + InferStr + LatticeValue,
                 V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
                 &self,
                 a_id: V,
                 b: T)
                 -> ures;
    fn t_sub_var<T:Clone + InferStr + LatticeValue,
                 V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
                 &self,
                 a: T,
                 b_id: V)
                 -> ures;
    fn merge_bnd<T:Clone + InferStr + LatticeValue>(
                 &self,
                 a: &Bound<T>,
                 b: &Bound<T>,
                 lattice_op: LatticeOp<T>)
                 -> cres<Bound<T>>;
    fn set_var_to_merged_bounds<T:Clone + InferStr + LatticeValue,
                                V:Clone+Eq+ToStr+Vid+UnifyVid<Bounds<T>>>(
                                &self,
                                v_id: V,
                                a: &Bounds<T>,
                                b: &Bounds<T>,
                                rank: uint)
                                -> ures;
    fn bnds<T:Clone + InferStr + LatticeValue>(
            &self,
            a: &Bound<T>,
            b: &Bound<T>)
            -> ures;
}

impl<'f> CombineFieldsLatticeMethods for CombineFields<'f> {
    fn var_sub_var<T:Clone + InferStr + LatticeValue,
                   V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
                   &self,
                   a_id: V,
                   b_id: V)
                   -> ures {
        /*!
         *
         * Make one variable a subtype of another variable.  This is a
         * subtle and tricky process, as described in detail at the
         * top of infer.rs*/

        // Need to make sub_id a subtype of sup_id.
        let node_a = self.infcx.get(a_id);
        let node_b = self.infcx.get(b_id);
        let a_id = node_a.root.clone();
        let b_id = node_b.root.clone();
        let a_bounds = node_a.possible_types.clone();
        let b_bounds = node_b.possible_types.clone();

        debug!("vars({}={} <: {}={})",
               a_id.to_str(), a_bounds.inf_str(self.infcx),
               b_id.to_str(), b_bounds.inf_str(self.infcx));

        if a_id == b_id { return uok(); }

        // If both A's UB and B's LB have already been bound to types,
        // see if we can make those types subtypes.
        match (&a_bounds.ub, &b_bounds.lb) {
            (&Some(ref a_ub), &Some(ref b_lb)) => {
                let r = self.infcx.try(
                    || LatticeValue::sub(self, a_ub, b_lb));
                match r {
                    Ok(()) => {
                        return Ok(());
                    }
                    Err(_) => { /*fallthrough */ }
                }
            }
            _ => { /*fallthrough*/ }
        }

        // Otherwise, we need to merge A and B so as to guarantee that
        // A remains a subtype of B.  Actually, there are other options,
        // but that's the route we choose to take.

        let (new_root, new_rank) = self.infcx.unify(&node_a, &node_b);
        self.set_var_to_merged_bounds(new_root,
                                      &a_bounds, &b_bounds,
                                      new_rank)
    }

    /// make variable a subtype of T
    fn var_sub_t<T:Clone + InferStr + LatticeValue,
                 V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
                 &self,
                 a_id: V,
                 b: T)
                 -> ures {
        /*!
         *
         * Make a variable (`a_id`) a subtype of the concrete type `b` */

        let node_a = self.infcx.get(a_id);
        let a_id = node_a.root.clone();
        let a_bounds = &node_a.possible_types;
        let b_bounds = &Bounds { lb: None, ub: Some(b.clone()) };

        debug!("var_sub_t({}={} <: {})",
               a_id.to_str(),
               a_bounds.inf_str(self.infcx),
               b.inf_str(self.infcx));

        self.set_var_to_merged_bounds(
            a_id, a_bounds, b_bounds, node_a.rank)
    }

    fn t_sub_var<T:Clone + InferStr + LatticeValue,
                 V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
                 &self,
                 a: T,
                 b_id: V)
                 -> ures {
        /*!
         *
         * Make a concrete type (`a`) a subtype of the variable `b_id` */

        let a_bounds = &Bounds { lb: Some(a.clone()), ub: None };
        let node_b = self.infcx.get(b_id);
        let b_id = node_b.root.clone();
        let b_bounds = &node_b.possible_types;

        debug!("t_sub_var({} <: {}={})",
               a.inf_str(self.infcx),
               b_id.to_str(),
               b_bounds.inf_str(self.infcx));

        self.set_var_to_merged_bounds(
            b_id, a_bounds, b_bounds, node_b.rank)
    }

    fn merge_bnd<T:Clone + InferStr + LatticeValue>(
                 &self,
                 a: &Bound<T>,
                 b: &Bound<T>,
                 lattice_op: LatticeOp<T>)
                 -> cres<Bound<T>> {
        /*!
         *
         * Combines two bounds into a more general bound. */

        debug!("merge_bnd({},{})",
               a.inf_str(self.infcx),
               b.inf_str(self.infcx));
        let _r = indenter();

        match (a, b) {
            (&None,          &None) => Ok(None),
            (&Some(_),       &None) => Ok((*a).clone()),
            (&None,          &Some(_)) => Ok((*b).clone()),
            (&Some(ref v_a), &Some(ref v_b)) => {
                lattice_op(self, v_a, v_b).and_then(|v| Ok(Some(v)))
            }
        }
    }

    fn set_var_to_merged_bounds<T:Clone + InferStr + LatticeValue,
                                V:Clone+Eq+ToStr+Vid+UnifyVid<Bounds<T>>>(
                                &self,
                                v_id: V,
                                a: &Bounds<T>,
                                b: &Bounds<T>,
                                rank: uint)
                                -> ures {
        /*!
         *
         * Updates the bounds for the variable `v_id` to be the intersection
         * of `a` and `b`.  That is, the new bounds for `v_id` will be
         * a bounds c such that:
         *    c.ub <: a.ub
         *    c.ub <: b.ub
         *    a.lb <: c.lb
         *    b.lb <: c.lb
         * If this cannot be achieved, the result is failure. */

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

        debug!("merge({},{},{})",
               v_id.to_str(),
               a.inf_str(self.infcx),
               b.inf_str(self.infcx));
        let _indent = indenter();

        // First, relate the lower/upper bounds of A and B.
        // Note that these relations *must* hold for us
        // to be able to merge A and B at all, and relating
        // them explicitly gives the type inferencer more
        // information and helps to produce tighter bounds
        // when necessary.
        let () = if_ok!(self.bnds(&a.lb, &b.ub));
        let () = if_ok!(self.bnds(&b.lb, &a.ub));
        let ub = if_ok!(self.merge_bnd(&a.ub, &b.ub, LatticeValue::glb));
        let lb = if_ok!(self.merge_bnd(&a.lb, &b.lb, LatticeValue::lub));
        let bounds = Bounds { lb: lb, ub: ub };
        debug!("merge({}): bounds={}",
               v_id.to_str(),
               bounds.inf_str(self.infcx));

        // the new bounds must themselves
        // be relatable:
        let () = if_ok!(self.bnds(&bounds.lb, &bounds.ub));
        self.infcx.set(v_id, Root(bounds, rank));
        uok()
    }

    fn bnds<T:Clone + InferStr + LatticeValue>(&self,
                                               a: &Bound<T>,
                                               b: &Bound<T>)
                                               -> ures {
        debug!("bnds({} <: {})", a.inf_str(self.infcx),
               b.inf_str(self.infcx));
        let _r = indenter();

        match (a, b) {
            (&None, &None) |
            (&Some(_), &None) |
            (&None, &Some(_)) => {
                uok()
            }
            (&Some(ref t_a), &Some(ref t_b)) => {
                LatticeValue::sub(self, t_a, t_b)
            }
        }
    }
}

// ______________________________________________________________________
// Lattice operations on variables
//
// This is common code used by both LUB and GLB to compute the LUB/GLB
// for pairs of variables or for variables and values.

pub trait LatticeDir {
    fn combine_fields<'a>(&'a self) -> CombineFields<'a>;
    fn bnd<T:Clone>(&self, b: &Bounds<T>) -> Option<T>;
    fn with_bnd<T:Clone>(&self, b: &Bounds<T>, t: T) -> Bounds<T>;
}

pub trait TyLatticeDir {
    fn ty_bot(&self, t: ty::t) -> cres<ty::t>;
}

impl<'f> LatticeDir for Lub<'f> {
    fn combine_fields<'a>(&'a self) -> CombineFields<'a> { *self.get_ref() }
    fn bnd<T:Clone>(&self, b: &Bounds<T>) -> Option<T> { b.ub.clone() }
    fn with_bnd<T:Clone>(&self, b: &Bounds<T>, t: T) -> Bounds<T> {
        Bounds { ub: Some(t), ..(*b).clone() }
    }
}

impl<'f> TyLatticeDir for Lub<'f> {
    fn ty_bot(&self, t: ty::t) -> cres<ty::t> {
        Ok(t)
    }
}

impl<'f> LatticeDir for Glb<'f> {
    fn combine_fields<'a>(&'a self) -> CombineFields<'a> { *self.get_ref() }
    fn bnd<T:Clone>(&self, b: &Bounds<T>) -> Option<T> { b.lb.clone() }
    fn with_bnd<T:Clone>(&self, b: &Bounds<T>, t: T) -> Bounds<T> {
        Bounds { lb: Some(t), ..(*b).clone() }
    }
}

impl<'f> TyLatticeDir for Glb<'f> {
    fn ty_bot(&self, _t: ty::t) -> cres<ty::t> {
        Ok(ty::mk_bot())
    }
}

pub fn super_lattice_tys<L:LatticeDir+TyLatticeDir+Combine>(this: &L,
                                                            a: ty::t,
                                                            b: ty::t)
                                                            -> cres<ty::t> {
    debug!("{}.lattice_tys({}, {})", this.tag(),
           a.inf_str(this.infcx()),
           b.inf_str(this.infcx()));

    if a == b {
        return Ok(a);
    }

    let tcx = this.infcx().tcx;

    match (&ty::get(a).sty, &ty::get(b).sty) {
        (&ty::ty_bot, _) => { return this.ty_bot(b); }
        (_, &ty::ty_bot) => { return this.ty_bot(a); }

        (&ty::ty_infer(TyVar(a_id)), &ty::ty_infer(TyVar(b_id))) => {
            let r = if_ok!(lattice_vars(this, a_id, b_id,
                                        |x, y| this.tys(*x, *y)));
            return match r {
                VarResult(v) => Ok(ty::mk_var(tcx, v)),
                ValueResult(t) => Ok(t)
            };
        }

        (&ty::ty_infer(TyVar(a_id)), _) => {
            return lattice_var_and_t(this, a_id, &b,
                                     |x, y| this.tys(*x, *y));
        }

        (_, &ty::ty_infer(TyVar(b_id))) => {
            return lattice_var_and_t(this, b_id, &a,
                                     |x, y| this.tys(*x, *y));
        }

        _ => {
            return super_tys(this, a, b);
        }
    }
}

pub type LatticeDirOp<'a, T> = 'a |a: &T, b: &T| -> cres<T>;

#[deriving(Clone)]
pub enum LatticeVarResult<V,T> {
    VarResult(V),
    ValueResult(T)
}

/**
 * Computes the LUB or GLB of two bounded variables.  These could be any
 * sort of variables, but in the comments on this function I'll assume
 * we are doing an LUB on two type variables.
 *
 * This computation can be done in one of two ways:
 *
 * - If both variables have an upper bound, we may just compute the
 *   LUB of those bounds and return that, in which case we are
 *   returning a type.  This is indicated with a `ValueResult` return.
 *
 * - If the variables do not both have an upper bound, we will unify
 *   the variables and return the unified variable, in which case the
 *   result is a variable.  This is indicated with a `VarResult`
 *   return. */
pub fn lattice_vars<L:LatticeDir + Combine,
                    T:Clone + InferStr + LatticeValue,
                    V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
    this: &L,                           // defines whether we want LUB or GLB
    a_vid: V,                          // first variable
    b_vid: V,                          // second variable
    lattice_dir_op: LatticeDirOp<T>)    // LUB or GLB operation on types
    -> cres<LatticeVarResult<V,T>> {
    let nde_a = this.infcx().get(a_vid);
    let nde_b = this.infcx().get(b_vid);
    let a_vid = nde_a.root.clone();
    let b_vid = nde_b.root.clone();
    let a_bounds = &nde_a.possible_types;
    let b_bounds = &nde_b.possible_types;

    debug!("{}.lattice_vars({}={} <: {}={})",
           this.tag(),
           a_vid.to_str(), a_bounds.inf_str(this.infcx()),
           b_vid.to_str(), b_bounds.inf_str(this.infcx()));

    // Same variable: the easy case.
    if a_vid == b_vid {
        return Ok(VarResult(a_vid));
    }

    // If both A and B have an UB type, then we can just compute the
    // LUB of those types:
    let (a_bnd, b_bnd) = (this.bnd(a_bounds), this.bnd(b_bounds));
    match (a_bnd, b_bnd) {
        (Some(ref a_ty), Some(ref b_ty)) => {
            match this.infcx().try(|| lattice_dir_op(a_ty, b_ty) ) {
                Ok(t) => return Ok(ValueResult(t)),
                Err(_) => { /*fallthrough */ }
            }
        }
        _ => {/*fallthrough*/}
    }

    // Otherwise, we need to merge A and B into one variable.  We can
    // then use either variable as an upper bound:
    let cf = this.combine_fields();
    cf.var_sub_var(a_vid.clone(), b_vid.clone()).then(|| {
        Ok(VarResult(a_vid.clone()))
    })
}

pub fn lattice_var_and_t<L:LatticeDir + Combine,
                         T:Clone + InferStr + LatticeValue,
                         V:Clone + Eq + ToStr + Vid + UnifyVid<Bounds<T>>>(
    this: &L,
    a_id: V,
    b: &T,
    lattice_dir_op: LatticeDirOp<T>)
    -> cres<T> {
    let nde_a = this.infcx().get(a_id);
    let a_id = nde_a.root.clone();
    let a_bounds = &nde_a.possible_types;

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    debug!("{}.lattice_var_and_t({}={} <: {})",
           this.tag(),
           a_id.to_str(),
           a_bounds.inf_str(this.infcx()),
           b.inf_str(this.infcx()));

    match this.bnd(a_bounds) {
        Some(ref a_bnd) => {
            // If a has an upper bound, return the LUB(a.ub, b)
            debug!("bnd=Some({})", a_bnd.inf_str(this.infcx()));
            lattice_dir_op(a_bnd, b)
        }
        None => {
            // If a does not have an upper bound, make b the upper bound of a
            // and then return b.
            debug!("bnd=None");
            let a_bounds = this.with_bnd(a_bounds, (*b).clone());
            this.combine_fields().bnds(&a_bounds.lb, &a_bounds.ub).then(|| {
                this.infcx().set(a_id.clone(),
                                 Root(a_bounds.clone(), nde_a.rank));
                Ok((*b).clone())
            })
        }
    }
}

// ___________________________________________________________________________
// Random utility functions used by LUB/GLB when computing LUB/GLB of
// fn types

pub fn var_ids<T:Combine>(this: &T,
                          map: &HashMap<ty::BoundRegion, ty::Region>)
                          -> Vec<RegionVid> {
    map.iter().map(|(_, r)| match *r {
            ty::ReInfer(ty::ReVar(r)) => { r }
            r => {
                this.infcx().tcx.sess.span_bug(
                    this.trace().origin.span(),
                    format!("found non-region-vid: {:?}", r));
            }
        }).collect()
}

pub fn is_var_in_set(new_vars: &[RegionVid], r: ty::Region) -> bool {
    match r {
        ty::ReInfer(ty::ReVar(ref v)) => new_vars.iter().any(|x| x == v),
        _ => false
    }
}
