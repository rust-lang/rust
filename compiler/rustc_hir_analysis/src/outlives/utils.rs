use rustc_infer::infer::outlives::components::{push_outlives_components, Component};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind};
use rustc_middle::ty::{self, Region, Ty, TyCtxt};
use rustc_span::Span;
use smallvec::smallvec;
use std::collections::BTreeMap;

/// Tracks the `T: 'a` or `'a: 'a` predicates that we have inferred
/// must be added to the struct header.
pub(crate) type RequiredPredicates<'tcx> =
    BTreeMap<ty::OutlivesPredicate<GenericArg<'tcx>, ty::Region<'tcx>>, Span>;

/// Given a requirement `T: 'a` or `'b: 'a`, deduce the
/// outlives_component and add it to `required_predicates`
pub(crate) fn insert_outlives_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: GenericArg<'tcx>,
    outlived_region: Region<'tcx>,
    span: Span,
    required_predicates: &mut RequiredPredicates<'tcx>,
) {
    // If the `'a` region is bound within the field type itself, we
    // don't want to propagate this constraint to the header.
    if !is_free_region(outlived_region) {
        return;
    }

    match kind.unpack() {
        GenericArgKind::Type(ty) => {
            // `T: 'outlived_region` for some type `T`
            // But T could be a lot of things:
            // e.g., if `T = &'b u32`, then `'b: 'outlived_region` is
            // what we want to add.
            //
            // Or if within `struct Foo<U>` you had `T = Vec<U>`, then
            // we would want to add `U: 'outlived_region`
            let mut components = smallvec![];
            push_outlives_components(tcx, ty, &mut components);
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
                        // u32`. Decomposing `&'b u32` into
                        // components would yield `'b`, and we add the
                        // where clause that `'b: 'a`.
                        insert_outlives_predicate(
                            tcx,
                            r.into(),
                            outlived_region,
                            span,
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
                        // Vec<U>`. Decomposing `Vec<U>` into
                        // components would yield `U`, and we add the
                        // where clause that `U: 'a`.
                        let ty: Ty<'tcx> = param_ty.to_ty(tcx);
                        required_predicates
                            .entry(ty::OutlivesPredicate(ty.into(), outlived_region))
                            .or_insert(span);
                    }

                    Component::Alias(alias_ty) => {
                        // This would either arise from something like:
                        //
                        // ```
                        // struct Foo<'a, T: Iterator> {
                        //    x:  &'a <T as Iterator>::Item
                        // }
                        // ```
                        //
                        // or:
                        //
                        // ```rust
                        // type Opaque<T> = impl Sized;
                        // fn defining<T>() -> Opaque<T> {}
                        // struct Ss<'a, T>(&'a Opaque<T>);
                        // ```
                        //
                        // Here we want to add an explicit `where <T as Iterator>::Item: 'a`
                        // or `Opaque<T>: 'a` depending on the alias kind.
                        let ty = alias_ty.to_ty(tcx);
                        required_predicates
                            .entry(ty::OutlivesPredicate(ty.into(), outlived_region))
                            .or_insert(span);
                    }

                    Component::EscapingAlias(_) => {
                        // As above, but the projection involves
                        // late-bound regions. Therefore, the WF
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

        GenericArgKind::Lifetime(r) => {
            if !is_free_region(r) {
                return;
            }
            required_predicates.entry(ty::OutlivesPredicate(kind, outlived_region)).or_insert(span);
        }

        GenericArgKind::Const(_) => {
            // Generic consts don't impose any constraints.
        }
    }
}

fn is_free_region(region: Region<'_>) -> bool {
    // First, screen for regions that might appear in a type header.
    match *region {
        // These correspond to `T: 'a` relationships:
        //
        //     struct Foo<'a, T> {
        //         field: &'a T, // this would generate a ReEarlyBound referencing `'a`
        //     }
        //
        // We care about these, so fall through.
        ty::ReEarlyBound(_) => true,

        // These correspond to `T: 'static` relationships which can be
        // rather surprising.
        //
        //     struct Foo<'a, T> {
        //         field: &'static T, // this would generate a ReStatic
        //     }
        ty::ReStatic => false,

        // Late-bound regions can appear in `fn` types:
        //
        //     struct Foo<T> {
        //         field: for<'b> fn(&'b T) // e.g., 'b here
        //     }
        //
        // The type above might generate a `T: 'b` bound, but we can
        // ignore it. We can't put it on the struct header anyway.
        ty::ReLateBound(..) => false,

        ty::ReError(_) => false,

        // These regions don't appear in types from type declarations:
        ty::ReErased | ty::ReVar(..) | ty::RePlaceholder(..) | ty::ReFree(..) => {
            bug!("unexpected region in outlives inference: {:?}", region);
        }
    }
}
