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
use middle::typeck::infer::*;
use middle::typeck::infer::combine::*;
use middle::typeck::infer::glb::Glb;
use middle::typeck::infer::lub::Lub;
use util::ppaux::Repr;

use std::collections::HashMap;

pub trait LatticeDir {
    // Relates the bottom type to `t` and returns LUB(t, _|_) or
    // GLB(t, _|_) as appropriate.
    fn ty_bot(&self, t: ty::t) -> cres<ty::t>;

    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    fn relate_bound<'a>(&'a self, v: ty::t, a: ty::t, b: ty::t) -> cres<()>;
}

impl<'a> LatticeDir for Lub<'a> {
    fn ty_bot(&self, t: ty::t) -> cres<ty::t> {
        Ok(t)
    }

    fn relate_bound<'a>(&'a self, v: ty::t, a: ty::t, b: ty::t) -> cres<()> {
        let sub = self.sub();
        try!(sub.tys(a, v));
        try!(sub.tys(b, v));
        Ok(())
    }
}

impl<'a> LatticeDir for Glb<'a> {
    fn ty_bot(&self, _: ty::t) -> cres<ty::t> {
        Ok(ty::mk_bot())
    }

    fn relate_bound<'a>(&'a self, v: ty::t, a: ty::t, b: ty::t) -> cres<()> {
        let sub = self.sub();
        try!(sub.tys(v, a));
        try!(sub.tys(v, b));
        Ok(())
    }
}

pub fn super_lattice_tys<L:LatticeDir+Combine>(this: &L,
                                               a: ty::t,
                                               b: ty::t)
                                               -> cres<ty::t>
{
    debug!("{}.lattice_tys({}, {})",
           this.tag(),
           a.repr(this.infcx().tcx),
           b.repr(this.infcx().tcx));

    if a == b {
        return Ok(a);
    }

    let infcx = this.infcx();
    let a = infcx.type_variables.borrow().replace_if_possible(a);
    let b = infcx.type_variables.borrow().replace_if_possible(b);
    match (&ty::get(a).sty, &ty::get(b).sty) {
        (&ty::ty_bot, _) => { this.ty_bot(b) }
        (_, &ty::ty_bot) => { this.ty_bot(a) }

        (&ty::ty_infer(TyVar(..)), _) |
        (_, &ty::ty_infer(TyVar(..))) => {
            let v = infcx.next_ty_var();
            try!(this.relate_bound(v, a, b));
            Ok(v)
        }

        _ => {
            super_tys(this, a, b)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
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
