use rustc_hir::def_id::DefId;
use rustc_infer::infer::outlives::components::{push_outlives_components, Component};
use rustc_middle::ty::subst::{GenericArg, GenericArgKind};
use rustc_middle::ty::{self, Region, Ty, TyCtxt};
use rustc_span::Span;
use smallvec::{smallvec, SmallVec};
use std::collections::BTreeMap;

pub(crate) type DefStack = SmallVec<[(DefId, Span); 3]>;

/// Tracks the `T: 'a` or `'a: 'a` predicates that we have inferred
/// must be added to the struct header.
pub(crate) type RequiredPredicates<'tcx> =
    BTreeMap<ty::OutlivesPredicate<GenericArg<'tcx>, ty::Region<'tcx>>, DefStack>;

/// Given a requirement `T: 'a` or `'b: 'a`, deduce the
/// outlives_component and add it to `required_predicates`
pub(crate) fn insert_outlives_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: GenericArg<'tcx>,
    outlived_region: Region<'tcx>,
    self_did: DefId,
    span: Span,
    required_predicates: &mut RequiredPredicates<'tcx>,
    stack: Option<&DefStack>,
) {
    // If the `'a` region is bound within the field type itself, we
    // don't want to propagate this constraint to the header.
    if !is_free_region(outlived_region) {
        return;
    }

    let insert_and_stack_push = |rpred: &mut RequiredPredicates<'tcx>, arg: GenericArg<'tcx>| {
        rpred
            .entry(ty::OutlivesPredicate(arg, outlived_region))
            .and_modify(|did_stack| did_stack.push((self_did, span)))
            .or_insert_with(move || {
                let mut stack = stack.cloned().unwrap_or_else(|| smallvec![]);
                stack.push((self_did, span));
                stack
            });
    };

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
                        // u32`.  Decomposing `&'b u32` into
                        // components would yield `'b`, and we add the
                        // where clause that `'b: 'a`.
                        insert_outlives_predicate(
                            tcx,
                            r.into(),
                            outlived_region,
                            self_did,
                            span,
                            required_predicates,
                            stack,
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
                        insert_and_stack_push(required_predicates, ty.into());
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
                        insert_and_stack_push(required_predicates, ty.into());
                    }

                    Component::Opaque(def_id, substs) => {
                        // This would arise from something like:
                        //
                        // ```rust
                        // type Opaque<T> = impl Sized;
                        // fn defining<T>() -> Opaque<T> {}
                        // struct Ss<'a, T>(&'a Opaque<T>);
                        // ```
                        //
                        // Here we want to have an implied bound `Opaque<T>: 'a`

                        let ty = tcx.mk_opaque(def_id, substs);
                        insert_and_stack_push(required_predicates, ty.into());
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

        GenericArgKind::Lifetime(r) => {
            if !is_free_region(r) {
                return;
            }
            insert_and_stack_push(required_predicates, kind);
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
        // ignore it.  We can't put it on the struct header anyway.
        ty::ReLateBound(..) => false,

        // These regions don't appear in types from type declarations:
        ty::ReErased | ty::ReVar(..) | ty::RePlaceholder(..) | ty::ReFree(..) => {
            bug!("unexpected region in outlives inference: {:?}", region);
        }
    }
}
