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

use super::InferCtxt;
use super::type_variable::TypeVariableOrigin;

use traits::ObligationCause;
use ty::TyVar;
use ty::{self, Ty};
use ty::relate::{RelateResult, TypeRelation};

pub trait LatticeDir<'f, 'gcx: 'f+'tcx, 'tcx: 'f> : TypeRelation<'f, 'gcx, 'tcx> {
    fn infcx(&self) -> &'f InferCtxt<'f, 'gcx, 'tcx>;

    fn cause(&self) -> &ObligationCause<'tcx>;

    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()>;
}

pub fn super_lattice_tys<'a, 'gcx, 'tcx, L>(this: &mut L,
                                            a: Ty<'tcx>,
                                            b: Ty<'tcx>)
                                            -> RelateResult<'tcx, Ty<'tcx>>
    where L: LatticeDir<'a, 'gcx, 'tcx>, 'gcx: 'a+'tcx, 'tcx: 'a
{
    debug!("{}.lattice_tys({:?}, {:?})",
           this.tag(),
           a,
           b);

    if a == b {
        return Ok(a);
    }

    let infcx = this.infcx();
    let a = infcx.type_variables.borrow_mut().replace_if_possible(a);
    let b = infcx.type_variables.borrow_mut().replace_if_possible(b);
    match (&a.sty, &b.sty) {
        (&ty::TyInfer(TyVar(..)), &ty::TyInfer(TyVar(..)))
            if infcx.type_var_diverges(a) && infcx.type_var_diverges(b) => {
            let v = infcx.next_diverging_ty_var(
                TypeVariableOrigin::LatticeVariable(this.cause().span));
            this.relate_bound(v, a, b)?;
            Ok(v)
        }

        (&ty::TyInfer(TyVar(..)), _) |
        (_, &ty::TyInfer(TyVar(..))) => {
            let v = infcx.next_ty_var(TypeVariableOrigin::LatticeVariable(this.cause().span));
            this.relate_bound(v, a, b)?;
            Ok(v)
        }

        _ => {
            infcx.super_combine_tys(this, a, b)
        }
    }
}
