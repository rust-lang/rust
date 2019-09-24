use rustc::ty::outlives::Component;
use rustc::ty::subst::{Kind, UnpackedKind};
use rustc::ty::{self, Region, RegionKind, Ty, TyCtxt};
use smallvec::smallvec;
use std::collections::BTreeSet;

/// Tracks the `T: 'a` or `'a: 'a` predicates that we have inferred
/// must be added to the struct header.
pub type RequiredPredicates<'tcx> = BTreeSet<ty::OutlivesPredicate<Kind<'tcx>, ty::Region<'tcx>>>;

/// Given a requirement `T: 'a` or `'b: 'a`, deduce the
/// outlives_component and add it to `required_predicates`
pub fn insert_outlives_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: Kind<'tcx>,
    outlived_region: Region<'tcx>,
    required_predicates: &mut RequiredPredicates<'tcx>,
) {
    // If the `'a` region is bound within the field type itself, we
    // don't want to propagate this constraint to the header.
    if !is_free_region(tcx, outlived_region) {
        return;
    }

    match kind.unpack() {
        UnpackedKind::Type(ty) => {
            // `T: 'outlived_region` for some type `T`
            // But T could be a lot of things:
            // e.g., if `T = &'b u32`, then `'b: 'outlived_region` is
            // what we want to add.
            //
            // Or if within `struct Foo<U>` you had `T = Vec<U>`, then
            // we would want to add `U: 'outlived_region`
            let mut components = smallvec![];
            tcx.push_outlives_components(ty, &mut components);
            for component in components {
                match component {
                    Component::Region(r) => {
                        // This would arise from something like:
                        //
                        // ```
                        // struct Foo<'a, 'b> {
                        //    x:  &'a &'b u32
                        // }
                        // ```
                        //
                        // Here `outlived_region = 'a` and `kind = &'b
                        // u32`.  Decomposing `&'b u32` into
                        // components would yield `'b`, and we add the
                        // where clause that `'b: 'a`.
                        insert_outlives_predicate(
                            tcx,
                            r.into(),
                            outlived_region,
                            required_predicates,
                        );
                    }

                    Component::Param(param_ty) => {
                        // param_ty: ty::ParamTy
                        // This would arise from something like:
                        //
                        // ```
                        // struct Foo<'a, U> {
                        //    x:  &'a Vec<U>
                        // }
                        // ```
                        //
                        // Here `outlived_region = 'a` and `kind =
                        // Vec<U>`.  Decomposing `Vec<U>` into
                        // components would yield `U`, and we add the
                        // where clause that `U: 'a`.
                        let ty: Ty<'tcx> = param_ty.to_ty(tcx);
                        required_predicates
                            .insert(ty::OutlivesPredicate(ty.into(), outlived_region));
                    }

                    Component::Projection(proj_ty) => {
                        // This would arise from something like:
                        //
                        // ```
                        // struct Foo<'a, T: Iterator> {
                        //    x:  &'a <T as Iterator>::Item
                        // }
                        // ```
                        //
                        // Here we want to add an explicit `where <T as Iterator>::Item: 'a`.
                        let ty: Ty<'tcx> = tcx.mk_projection(proj_ty.item_def_id, proj_ty.substs);
                        required_predicates
                            .insert(ty::OutlivesPredicate(ty.into(), outlived_region));
                    }

                    Component::EscapingProjection(_) => {
                        // As above, but the projection involves
                        // late-bound regions.  Therefore, the WF
                        // requirement is not checked in type definition
                        // but at fn call site, so ignore it.
                        //
                        // ```
                        // struct Foo<'a, T: Iterator> {
                        //    x: for<'b> fn(<&'b T as Iterator>::Item)
                        //              //  ^^^^^^^^^^^^^^^^^^^^^^^^^
                        // }
                        // ```
                        //
                        // Since `'b` is not in scope on `Foo`, can't
                        // do anything here, ignore it.
                    }

                    Component::UnresolvedInferenceVariable(_) => bug!("not using infcx"),
                }
            }
        }

        UnpackedKind::Lifetime(r) => {
            if !is_free_region(tcx, r) {
                return;
            }
            required_predicates.insert(ty::OutlivesPredicate(kind, outlived_region));
        }

        UnpackedKind::Const(_) => {
            // Generic consts don't impose any constraints.
        }
    }
}

fn is_free_region(tcx: TyCtxt<'_>, region: Region<'_>) -> bool {
    // First, screen for regions that might appear in a type header.
    match region {
        // These correspond to `T: 'a` relationships:
        //
        //     struct Foo<'a, T> {
        //         field: &'a T, // this would generate a ReEarlyBound referencing `'a`
        //     }
        //
        // We care about these, so fall through.
        RegionKind::ReEarlyBound(_) => true,

        // These correspond to `T: 'static` relationships which can be
        // rather surprising. We are therefore putting this behind a
        // feature flag:
        //
        //     struct Foo<'a, T> {
        //         field: &'static T, // this would generate a ReStatic
        //     }
        RegionKind::ReStatic => {
            tcx.sess
               .features_untracked()
               .infer_static_outlives_requirements
        }

        // Late-bound regions can appear in `fn` types:
        //
        //     struct Foo<T> {
        //         field: for<'b> fn(&'b T) // e.g., 'b here
        //     }
        //
        // The type above might generate a `T: 'b` bound, but we can
        // ignore it.  We can't put it on the struct header anyway.
        RegionKind::ReLateBound(..) => false,

        // These regions don't appear in types from type declarations:
        RegionKind::ReEmpty
        | RegionKind::ReErased
        | RegionKind::ReClosureBound(..)
        | RegionKind::ReScope(..)
        | RegionKind::ReVar(..)
        | RegionKind::RePlaceholder(..)
        | RegionKind::ReFree(..) => {
            bug!("unexpected region in outlives inference: {:?}", region);
        }
    }
}
