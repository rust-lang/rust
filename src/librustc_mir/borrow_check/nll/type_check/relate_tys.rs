// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::constraints::OutlivesConstraint;
use borrow_check::nll::type_check::{BorrowCheckContext, Locations};
use rustc::infer::nll_relate::{TypeRelating, TypeRelatingDelegate};
use rustc::infer::{InferCtxt, NLLRegionVariableOrigin};
use rustc::mir::{ConstraintCategory, UserTypeAnnotation};
use rustc::traits::query::Fallible;
use rustc::ty::relate::TypeRelation;
use rustc::ty::subst::{Subst, UserSelfTy, UserSubsts};
use rustc::ty::{self, Ty, TypeFoldable};
use syntax_pos::DUMMY_SP;

/// Adds sufficient constraints to ensure that `a <: b`.
pub(super) fn sub_types<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
    locations: Locations,
    category: ConstraintCategory,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<()> {
    debug!("sub_types(a={:?}, b={:?}, locations={:?})", a, b, locations);
    TypeRelating::new(
        infcx,
        NllTypeRelatingDelegate::new(infcx, borrowck_context, locations, category),
        ty::Variance::Covariant,
    ).relate(&a, &b)?;
    Ok(())
}

/// Adds sufficient constraints to ensure that `a == b`.
pub(super) fn eq_types<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
    locations: Locations,
    category: ConstraintCategory,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<()> {
    debug!("eq_types(a={:?}, b={:?}, locations={:?})", a, b, locations);
    TypeRelating::new(
        infcx,
        NllTypeRelatingDelegate::new(infcx, borrowck_context, locations, category),
        ty::Variance::Invariant,
    ).relate(&a, &b)?;
    Ok(())
}

/// Adds sufficient constraints to ensure that `a <: b`, where `b` is
/// a user-given type (which means it may have canonical variables
/// encoding things like `_`).
pub(super) fn relate_type_and_user_type<'tcx>(
    infcx: &InferCtxt<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    v: ty::Variance,
    user_ty: UserTypeAnnotation<'tcx>,
    locations: Locations,
    category: ConstraintCategory,
    borrowck_context: Option<&mut BorrowCheckContext<'_, 'tcx>>,
) -> Fallible<Ty<'tcx>> {
    debug!(
        "relate_type_and_user_type(a={:?}, v={:?}, b={:?}, locations={:?})",
        a, v, user_ty, locations
    );

    // The `TypeRelating` code assumes that the "canonical variables"
    // appear in the "a" side, so flip `Contravariant` ambient
    // variance to get the right relationship.
    let v1 = ty::Contravariant.xform(v);

    let mut type_relating = TypeRelating::new(
        infcx,
        NllTypeRelatingDelegate::new(infcx, borrowck_context, locations, category),
        v1,
    );

    match user_ty {
        UserTypeAnnotation::Ty(canonical_ty) => {
            let (ty, _) =
                infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonical_ty);
            type_relating.relate(&ty, &a)?;
            Ok(ty)
        }
        UserTypeAnnotation::FnDef(def_id, canonical_substs) => {
            let (
                UserSubsts {
                    substs,
                    user_self_ty,
                },
                _,
            ) = infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonical_substs);
            let ty = infcx.tcx.mk_fn_def(def_id, substs);

            type_relating.relate(&ty, &a)?;

            if let Some(UserSelfTy {
                impl_def_id,
                self_ty,
            }) = user_self_ty
            {
                let impl_self_ty = infcx.tcx.type_of(impl_def_id);
                let impl_self_ty = impl_self_ty.subst(infcx.tcx, &substs);

                // There may be type variables in `substs` and hence
                // in `impl_self_ty`, but they should all have been
                // resolved to some fixed value during the first call
                // to `relate`, above. Therefore, if we use
                // `resolve_type_vars_if_possible` we should get to
                // something without type variables. This is important
                // because the `b` type in `relate_with_variance`
                // below is not permitted to have inference variables.
                let impl_self_ty = infcx.resolve_type_vars_if_possible(&impl_self_ty);
                assert!(!impl_self_ty.has_infer_types());

                type_relating.relate_with_variance(
                    ty::Variance::Invariant,
                    &self_ty,
                    &impl_self_ty,
                )?;
            }

            Ok(ty)
        }
        UserTypeAnnotation::AdtDef(adt_def, canonical_substs) => {
            let (
                UserSubsts {
                    substs,
                    user_self_ty,
                },
                _,
            ) = infcx.instantiate_canonical_with_fresh_inference_vars(DUMMY_SP, &canonical_substs);

            // We don't extract adt-defs with a self-type.
            assert!(user_self_ty.is_none());

            let ty = infcx.tcx.mk_adt(adt_def, substs);
            type_relating.relate(&ty, &a)?;
            Ok(ty)
        }
    }
}

struct NllTypeRelatingDelegate<'me, 'bccx: 'me, 'gcx: 'tcx, 'tcx: 'bccx> {
    infcx: &'me InferCtxt<'me, 'gcx, 'tcx>,
    borrowck_context: Option<&'me mut BorrowCheckContext<'bccx, 'tcx>>,

    /// Where (and why) is this relation taking place?
    locations: Locations,

    /// What category do we assign the resulting `'a: 'b` relationships?
    category: ConstraintCategory,
}

impl NllTypeRelatingDelegate<'me, 'bccx, 'gcx, 'tcx> {
    fn new(
        infcx: &'me InferCtxt<'me, 'gcx, 'tcx>,
        borrowck_context: Option<&'me mut BorrowCheckContext<'bccx, 'tcx>>,
        locations: Locations,
        category: ConstraintCategory,
    ) -> Self {
        Self {
            infcx,
            borrowck_context,
            locations,
            category,
        }
    }
}

impl TypeRelatingDelegate<'tcx> for NllTypeRelatingDelegate<'_, '_, '_, 'tcx> {
    fn create_next_universe(&mut self) -> ty::UniverseIndex {
        self.infcx.create_next_universe()
    }

    fn next_existential_region_var(&mut self) -> ty::Region<'tcx> {
        let origin = NLLRegionVariableOrigin::Existential;
        self.infcx.next_nll_region_var(origin)
    }

    fn next_placeholder_region(&mut self, placeholder: ty::Placeholder) -> ty::Region<'tcx> {
        let origin = NLLRegionVariableOrigin::Placeholder(placeholder);
        if let Some(borrowck_context) = &mut self.borrowck_context {
            borrowck_context.placeholder_indices.insert(placeholder);
        }
        self.infcx.next_nll_region_var(origin)
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        self.infcx
            .next_nll_region_var_in_universe(NLLRegionVariableOrigin::Existential, universe)
    }

    fn push_outlives(&mut self, sup: ty::Region<'tcx>, sub: ty::Region<'tcx>) {
        if let Some(borrowck_context) = &mut self.borrowck_context {
            let sub = borrowck_context.universal_regions.to_region_vid(sub);
            let sup = borrowck_context.universal_regions.to_region_vid(sup);
            borrowck_context
                .constraints
                .outlives_constraints
                .push(OutlivesConstraint {
                    sup,
                    sub,
                    locations: self.locations,
                    category: self.category,
                });
        }
    }
}
