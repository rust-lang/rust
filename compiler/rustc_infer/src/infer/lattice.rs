//! # Lattice variables
//!
//! Generic code for operating on [lattices] of inference variables
//! that are characterized by an upper- and lower-bound.
//!
//! The code is defined quite generically so that it can be
//! applied both to type variables, which represent types being inferred,
//! and fn variables, which represent function types being inferred.
//! (It may eventually be applied to their types as well.)
//! In some cases, the functions are also generic with respect to the
//! operation on the lattice (GLB vs LUB).
//!
//! ## Note
//!
//! Although all the functions are generic, for simplicity, comments in the source code
//! generally refer to type variables and the LUB operation.
//!
//! [lattices]: https://en.wikipedia.org/wiki/Lattice_(order)

use super::combine::ObligationEmittingRelation;
use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::InferCtxt;

use crate::traits::ObligationCause;
use rustc_middle::ty::relate::RelateResult;
use rustc_middle::ty::TyVar;
use rustc_middle::ty::{self, Ty};

/// Trait for returning data about a lattice, and for abstracting
/// over the "direction" of the lattice operation (LUB/GLB).
///
/// GLB moves "down" the lattice (to smaller values); LUB moves
/// "up" the lattice (to bigger values).
pub trait LatticeDir<'f, 'tcx>: ObligationEmittingRelation<'tcx> {
    fn infcx(&self) -> &'f InferCtxt<'tcx>;

    fn cause(&self) -> &ObligationCause<'tcx>;

    fn define_opaque_types(&self) -> bool;

    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    //
    // Subtle hack: ordering *may* be significant here. This method
    // relates `v` to `a` first, which may help us to avoid unnecessary
    // type variable obligations. See caller for details.
    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()>;
}

/// Relates two types using a given lattice.
#[instrument(skip(this), level = "debug")]
pub fn super_lattice_tys<'a, 'tcx: 'a, L>(
    this: &mut L,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> RelateResult<'tcx, Ty<'tcx>>
where
    L: LatticeDir<'a, 'tcx>,
{
    debug!("{}", this.tag());

    if a == b {
        return Ok(a);
    }

    let infcx = this.infcx();

    let a = infcx.inner.borrow_mut().type_variables().replace_if_possible(a);
    let b = infcx.inner.borrow_mut().type_variables().replace_if_possible(b);

    match (a.kind(), b.kind()) {
        // If one side is known to be a variable and one is not,
        // create a variable (`v`) to represent the LUB. Make sure to
        // relate `v` to the non-type-variable first (by passing it
        // first to `relate_bound`). Otherwise, we would produce a
        // subtype obligation that must then be processed.
        //
        // Example: if the LHS is a type variable, and RHS is
        // `Box<i32>`, then we current compare `v` to the RHS first,
        // which will instantiate `v` with `Box<i32>`. Then when `v`
        // is compared to the LHS, we instantiate LHS with `Box<i32>`.
        // But if we did in reverse order, we would create a `v <:
        // LHS` (or vice versa) constraint and then instantiate
        // `v`. This would require further processing to achieve same
        // end-result; in particular, this screws up some of the logic
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

        (
            &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
            &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
        ) if a_def_id == b_def_id => infcx.super_combine_tys(this, a, b),
        (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
        | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
            if this.define_opaque_types() && def_id.is_local() =>
        {
            this.register_obligations(
                infcx
                    .handle_opaque_type(a, b, this.a_is_expected(), this.cause(), this.param_env())?
                    .obligations,
            );
            Ok(a)
        }

        _ => infcx.super_combine_tys(this, a, b),
    }
}
