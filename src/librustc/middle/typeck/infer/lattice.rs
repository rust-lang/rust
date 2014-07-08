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
 * It may eventually be applied to their types as well, who knows.
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

use middle::ty::{RegionVid, TyVar};
use middle::ty;
use middle::typeck::infer::{ToUres};
use middle::typeck::infer::*;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use middle::typeck::infer::unify::*;
use middle::typeck::infer::sub::Sub;
use util::ppaux::Repr;

use std::collections::HashMap;

trait LatticeValue : Clone + Repr + PartialEq {
    fn sub(cf: CombineFields, a: &Self, b: &Self) -> ures;
    fn lub(cf: CombineFields, a: &Self, b: &Self) -> cres<Self>;
    fn glb(cf: CombineFields, a: &Self, b: &Self) -> cres<Self>;
}

pub type LatticeOp<'a, T> =
    |cf: CombineFields, a: &T, b: &T|: 'a -> cres<T>;

impl LatticeValue for ty::t {
    fn sub(cf: CombineFields, a: &ty::t, b: &ty::t) -> ures {
        Sub(cf).tys(*a, *b).to_ures()
    }

    fn lub(cf: CombineFields, a: &ty::t, b: &ty::t) -> cres<ty::t> {
        Lub(cf).tys(*a, *b)
    }

    fn glb(cf: CombineFields, a: &ty::t, b: &ty::t) -> cres<ty::t> {
        Glb(cf).tys(*a, *b)
    }
}

pub trait CombineFieldsLatticeMethods<T:LatticeValue, K:UnifyKey<Bounds<T>>> {
    /// make variable a subtype of variable
    fn var_sub_var(&self,
                   a_id: K,
                   b_id: K)
                   -> ures;

    /// make variable a subtype of T
    fn var_sub_t(&self,
                 a_id: K,
                 b: T)
                 -> ures;

    /// make T a subtype of variable
    fn t_sub_var(&self,
                 a: T,
                 b_id: K)
                 -> ures;

    fn set_var_to_merged_bounds(&self,
                                v_id: K,
                                a: &Bounds<T>,
                                b: &Bounds<T>,
                                rank: uint)
                                -> ures;
}

pub trait CombineFieldsLatticeMethods2<T:LatticeValue> {
    fn merge_bnd(&self,
                 a: &Bound<T>,
                 b: &Bound<T>,
                 lattice_op: LatticeOp<T>)
                 -> cres<Bound<T>>;

    fn bnds(&self, a: &Bound<T>, b: &Bound<T>) -> ures;
}

impl<'f,T:LatticeValue, K:UnifyKey<Bounds<T>>>
    CombineFieldsLatticeMethods<T,K> for CombineFields<'f>
{
    fn var_sub_var(&self,
                   a_id: K,
                   b_id: K)
                   -> ures
    {
        /*!
         * Make one variable a subtype of another variable.  This is a
         * subtle and tricky process, as described in detail at the
         * top of infer.rs.
         */

        let tcx = self.infcx.tcx;
        let table = UnifyKey::unification_table(self.infcx);

        // Need to make sub_id a subtype of sup_id.
        let node_a = table.borrow_mut().get(tcx, a_id);
        let node_b = table.borrow_mut().get(tcx, b_id);
        let a_id = node_a.key.clone();
        let b_id = node_b.key.clone();
        let a_bounds = node_a.value.clone();
        let b_bounds = node_b.value.clone();

        debug!("vars({}={} <: {}={})",
               a_id, a_bounds.repr(tcx),
               b_id, b_bounds.repr(tcx));

        if a_id == b_id { return Ok(()); }

        // If both A's UB and B's LB have already been bound to types,
        // see if we can make those types subtypes.
        match (&a_bounds.ub, &b_bounds.lb) {
            (&Some(ref a_ub), &Some(ref b_lb)) => {
                let r = self.infcx.try(
                    || LatticeValue::sub(self.clone(), a_ub, b_lb));
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

        let (new_root, new_rank) =
            table.borrow_mut().unify(tcx, &node_a, &node_b);
        self.set_var_to_merged_bounds(new_root,
                                      &a_bounds, &b_bounds,
                                      new_rank)
    }

    /// make variable a subtype of T
    fn var_sub_t(&self,
                 a_id: K,
                 b: T)
                 -> ures
    {
        /*!
         * Make a variable (`a_id`) a subtype of the concrete type `b`.
         */

        let tcx = self.infcx.tcx;
        let table = UnifyKey::unification_table(self.infcx);
        let node_a = table.borrow_mut().get(tcx, a_id);
        let a_id = node_a.key.clone();
        let a_bounds = &node_a.value;
        let b_bounds = &Bounds { lb: None, ub: Some(b.clone()) };

        debug!("var_sub_t({}={} <: {})",
               a_id,
               a_bounds.repr(self.infcx.tcx),
               b.repr(self.infcx.tcx));

        self.set_var_to_merged_bounds(
            a_id, a_bounds, b_bounds, node_a.rank)
    }

    fn t_sub_var(&self,
                 a: T,
                 b_id: K)
                 -> ures
    {
        /*!
         * Make a concrete type (`a`) a subtype of the variable `b_id`
         */

        let tcx = self.infcx.tcx;
        let table = UnifyKey::unification_table(self.infcx);
        let a_bounds = &Bounds { lb: Some(a.clone()), ub: None };
        let node_b = table.borrow_mut().get(tcx, b_id);
        let b_id = node_b.key.clone();
        let b_bounds = &node_b.value;

        debug!("t_sub_var({} <: {}={})",
               a.repr(self.infcx.tcx),
               b_id,
               b_bounds.repr(self.infcx.tcx));

        self.set_var_to_merged_bounds(
            b_id, a_bounds, b_bounds, node_b.rank)
    }

    fn set_var_to_merged_bounds(&self,
                                v_id: K,
                                a: &Bounds<T>,
                                b: &Bounds<T>,
                                rank: uint)
                                -> ures
    {
        /*!
         * Updates the bounds for the variable `v_id` to be the intersection
         * of `a` and `b`.  That is, the new bounds for `v_id` will be
         * a bounds c such that:
         *    c.ub <: a.ub
         *    c.ub <: b.ub
         *    a.lb <: c.lb
         *    b.lb <: c.lb
         * If this cannot be achieved, the result is failure.
         */

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

        let tcx = self.infcx.tcx;
        let table = UnifyKey::unification_table(self.infcx);

        debug!("merge({},{},{})",
               v_id,
               a.repr(self.infcx.tcx),
               b.repr(self.infcx.tcx));

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
               v_id,
               bounds.repr(self.infcx.tcx));

        // the new bounds must themselves
        // be relatable:
        let () = if_ok!(self.bnds(&bounds.lb, &bounds.ub));
        table.borrow_mut().set(tcx, v_id, Root(bounds, rank));
        Ok(())
    }
}

impl<'f,T:LatticeValue>
    CombineFieldsLatticeMethods2<T> for CombineFields<'f>
{
    fn merge_bnd(&self,
                 a: &Bound<T>,
                 b: &Bound<T>,
                 lattice_op: LatticeOp<T>)
                 -> cres<Bound<T>>
    {
        /*!
         * Combines two bounds into a more general bound.
         */

        debug!("merge_bnd({},{})",
               a.repr(self.infcx.tcx),
               b.repr(self.infcx.tcx));
        match (a, b) {
            (&None,          &None) => Ok(None),
            (&Some(_),       &None) => Ok((*a).clone()),
            (&None,          &Some(_)) => Ok((*b).clone()),
            (&Some(ref v_a), &Some(ref v_b)) => {
                lattice_op(self.clone(), v_a, v_b).and_then(|v| Ok(Some(v)))
            }
        }
    }

    fn bnds(&self,
            a: &Bound<T>,
            b: &Bound<T>)
            -> ures
    {
        debug!("bnds({} <: {})",
               a.repr(self.infcx.tcx),
               b.repr(self.infcx.tcx));

        match (a, b) {
            (&None, &None) |
            (&Some(_), &None) |
            (&None, &Some(_)) => {
                Ok(())
            }
            (&Some(ref t_a), &Some(ref t_b)) => {
                LatticeValue::sub(self.clone(), t_a, t_b)
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
    fn combine_fields<'a>(&'a self) -> CombineFields<'a> { self.get_ref().clone() }
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
    fn combine_fields<'a>(&'a self) -> CombineFields<'a> { self.get_ref().clone() }
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
    debug!("{}.lattice_tys({}, {})",
           this.tag(),
           a.repr(this.infcx().tcx),
           b.repr(this.infcx().tcx));

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

pub type LatticeDirOp<'a, T> = |a: &T, b: &T|: 'a -> cres<T>;

#[deriving(Clone)]
pub enum LatticeVarResult<K,T> {
    VarResult(K),
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
pub fn lattice_vars<L:LatticeDir+Combine,
                    T:LatticeValue,
                    K:UnifyKey<Bounds<T>>>(
    this: &L,                           // defines whether we want LUB or GLB
    a_vid: K,                           // first variable
    b_vid: K,                           // second variable
    lattice_dir_op: LatticeDirOp<T>)    // LUB or GLB operation on types
    -> cres<LatticeVarResult<K,T>>
{
    let tcx = this.infcx().tcx;
    let table = UnifyKey::unification_table(this.infcx());

    let node_a = table.borrow_mut().get(tcx, a_vid);
    let node_b = table.borrow_mut().get(tcx, b_vid);
    let a_vid = node_a.key.clone();
    let b_vid = node_b.key.clone();
    let a_bounds = &node_a.value;
    let b_bounds = &node_b.value;

    debug!("{}.lattice_vars({}={} <: {}={})",
           this.tag(),
           a_vid, a_bounds.repr(tcx),
           b_vid, b_bounds.repr(tcx));

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
    let () = try!(cf.var_sub_var(a_vid.clone(), b_vid.clone()));
    Ok(VarResult(a_vid.clone()))
}

pub fn lattice_var_and_t<L:LatticeDir+Combine,
                         T:LatticeValue,
                         K:UnifyKey<Bounds<T>>>(
    this: &L,
    a_id: K,
    b: &T,
    lattice_dir_op: LatticeDirOp<T>)
    -> cres<T>
{
    let tcx = this.infcx().tcx;
    let table = UnifyKey::unification_table(this.infcx());

    let node_a = table.borrow_mut().get(tcx, a_id);
    let a_id = node_a.key.clone();
    let a_bounds = &node_a.value;

    // The comments in this function are written for LUB, but they
    // apply equally well to GLB if you inverse upper/lower/sub/super/etc.

    debug!("{}.lattice_var_and_t({}={} <: {})",
           this.tag(),
           a_id,
           a_bounds.repr(this.infcx().tcx),
           b.repr(this.infcx().tcx));

    match this.bnd(a_bounds) {
        Some(ref a_bnd) => {
            // If a has an upper bound, return the LUB(a.ub, b)
            debug!("bnd=Some({})", a_bnd.repr(this.infcx().tcx));
            lattice_dir_op(a_bnd, b)
        }
        None => {
            // If a does not have an upper bound, make b the upper bound of a
            // and then return b.
            debug!("bnd=None");
            let a_bounds = this.with_bnd(a_bounds, (*b).clone());
            let () = try!(this.combine_fields().bnds(&a_bounds.lb,
                                                     &a_bounds.ub));
            table.borrow_mut().set(tcx,
                                   a_id.clone(),
                                   Root(a_bounds.clone(), node_a.rank));
            Ok((*b).clone())
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
                    format!("found non-region-vid: {:?}", r).as_slice());
            }
        }).collect()
}

pub fn is_var_in_set(new_vars: &[RegionVid], r: ty::Region) -> bool {
    match r {
        ty::ReInfer(ty::ReVar(ref v)) => new_vars.iter().any(|x| x == v),
        _ => false
    }
}
