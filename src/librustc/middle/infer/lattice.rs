// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Lattice Variables
//!
//! This file contains generic code for operating on inference variables
//! that are characterized by an upper- and lower-bound.  The logic and
//! reasoning is explained in detail in the large comment in `infer.rs`.
//!
//! The code in here is defined quite generically so that it can be
//! applied both to type variables, which represent types being inferred,
//! and fn variables, which represent function types being inferred.
//! It may eventually be applied to their types as well, who knows.
//! In some cases, the functions are also generic with respect to the
//! operation on the lattice (GLB vs LUB).
//!
//! Although all the functions are generic, we generally write the
//! comments in a way that is specific to type variables and the LUB
//! operation.  It's just easier that way.
//!
//! In general all of the functions are defined parametrically
//! over a `LatticeValue`, which is a value defined with respect to
//! a lattice.

use super::*;
use super::combine::*;
use super::glb::Glb;
use super::lub::Lub;

use middle::ty::{TyVar};
use middle::ty::{self, Ty};
use util::ppaux::Repr;

pub trait LatticeDir<'tcx> {
    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    fn relate_bound(&self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, ()>;
}

impl<'a, 'tcx> LatticeDir<'tcx> for Lub<'a, 'tcx> {
    fn relate_bound(&self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, ()> {
        let sub = self.sub();
        try!(sub.tys(a, v));
        try!(sub.tys(b, v));
        Ok(())
    }
}

impl<'a, 'tcx> LatticeDir<'tcx> for Glb<'a, 'tcx> {
    fn relate_bound(&self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> cres<'tcx, ()> {
        let sub = self.sub();
        try!(sub.tys(v, a));
        try!(sub.tys(v, b));
        Ok(())
    }
}

pub fn super_lattice_tys<'tcx, L:LatticeDir<'tcx>+Combine<'tcx>>(this: &L,
                                                                 a: Ty<'tcx>,
                                                                 b: Ty<'tcx>)
                                                                 -> cres<'tcx, Ty<'tcx>>
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
    match (&a.sty, &b.sty) {
        (&ty::ty_infer(TyVar(..)), &ty::ty_infer(TyVar(..)))
            if infcx.type_var_diverges(a) && infcx.type_var_diverges(b) => {
            let v = infcx.next_diverging_ty_var();
            try!(this.relate_bound(v, a, b));
            Ok(v)
        }

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
