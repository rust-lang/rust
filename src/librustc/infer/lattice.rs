//! # Lattice Variables
//!
//! This file contains generic code for operating on inference variables
//! that are characterized by an upper- and lower-bound. The logic and
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
//! operation. It's just easier that way.
//!
//! In general all of the functions are defined parametrically
//! over a `LatticeValue`, which is a value defined with respect to
//! a lattice.

use super::InferCtxt;
use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};

use crate::traits::ObligationCause;
use crate::ty::TyVar;
use crate::ty::{self, Ty};
use crate::ty::relate::{RelateResult, TypeRelation};

pub trait LatticeDir<'f, 'tcx>: TypeRelation<'tcx> {
    fn infcx(&self) -> &'f InferCtxt<'f, 'tcx>;

    fn cause(&self) -> &ObligationCause<'tcx>;

    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    //
    // Subtle hack: ordering *may* be significant here. This method
    // relates `v` to `a` first, which may help us to avoid unnecessary
    // type variable obligations. See caller for details.
    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()>;
}

pub fn super_lattice_tys<'a, 'tcx: 'a, L>(
    this: &mut L,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> RelateResult<'tcx, Ty<'tcx>>
where
    L: LatticeDir<'a, 'tcx>,
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
        // If one side is known to be a variable and one is not,
        // create a variable (`v`) to represent the LUB. Make sure to
        // relate `v` to the non-type-variable first (by passing it
        // first to `relate_bound`). Otherwise, we would produce a
        // subtype obligation that must then be processed.
        //
        // Example: if the LHS is a type variable, and RHS is
        // `Box<i32>`, then we current compare `v` to the RHS first,
        // which will instantiate `v` with `Box<i32>`.  Then when `v`
        // is compared to the LHS, we instantiate LHS with `Box<i32>`.
        // But if we did in reverse order, we would create a `v <:
        // LHS` (or vice versa) constraint and then instantiate
        // `v`. This would require further processing to achieve same
        // end-result; in partiular, this screws up some of the logic
        // in coercion, which expects LUB to figure out that the LHS
        // is (e.g.) `Box<i32>`. A more obvious solution might be to
        // iterate on the subtype obligations that are returned, but I
        // think this suffices. -nmatsakis
        (&ty::Infer(TyVar(..)), _) => {
            let v = infcx.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::LatticeVariable,
                span: this.cause().span,
            });
            this.relate_bound(v, b, a)?;
            Ok(v)
        }
        (_, &ty::Infer(TyVar(..))) => {
            let v = infcx.next_ty_var(TypeVariableOrigin {
                kind: TypeVariableOriginKind::LatticeVariable,
                span: this.cause().span,
            });
            this.relate_bound(v, a, b)?;
            Ok(v)
        }

        _ => {
            infcx.super_combine_tys(this, a, b)
        }
    }
}
