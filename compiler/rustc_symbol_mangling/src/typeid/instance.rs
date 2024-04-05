/// Adjusts an `Instance` prior to encoding.
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Instance, List, Ty, TyCtxt};
use rustc_trait_selection::traits;
use std::iter;

use crate::typeid;

/// Transform an instance where needed prior to encoding
///
/// Always:
///
/// * `drop_in_place::<T>` -> `drop_in_place::<dyn Drop>`
/// * `Trait::method::<dyn Foo + BarAuto>` -> `Trait::method::<dyn Trait>`
/// * `FnOnce::call_once<T>` (`VTableShim`) -> `FnOnce::call_once::<dyn FnOnce(Args) + Output=R>`
///
/// If `ERASE_SELF_TYPE` is set:
/// * `Trait::method::<T>` -> `Trait::method::<dyn Trait>`
pub fn transform<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut instance: Instance<'tcx>,
    options: typeid::Options,
) -> Instance<'tcx> {
    if (matches!(instance.def, ty::InstanceDef::Virtual(..))
        && Some(instance.def_id()) == tcx.lang_items().drop_in_place_fn())
        || matches!(instance.def, ty::InstanceDef::DropGlue(..))
    {
        // Adjust the type ids of DropGlues
        //
        // DropGlues may have indirect calls to one or more given types drop function. Rust allows
        // for types to be erased to any trait object and retains the drop function for the original
        // type, which means at the indirect call sites in DropGlues, when typeid_for_fnabi is
        // called a second time, it only has information after type erasure and it could be a call
        // on any arbitrary trait object. Normalize them to a synthesized Drop trait object, both on
        // declaration/definition, and during code generation at call sites so they have the same
        // type id and match.
        //
        // FIXME(rcvalle): This allows a drop call on any trait object to call the drop function of
        //   any other type.
        //
        let def_id = tcx
            .lang_items()
            .drop_trait()
            .unwrap_or_else(|| bug!("typeid_for_instance: couldn't get drop_trait lang item"));
        let predicate = ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef {
            def_id,
            args: List::empty(),
        });
        let predicates = tcx.mk_poly_existential_predicates(&[ty::Binder::dummy(predicate)]);
        let self_ty = Ty::new_dynamic(tcx, predicates, tcx.lifetimes.re_erased, ty::Dyn);
        instance.args = tcx.mk_args_trait(self_ty, List::empty());
    } else if let ty::InstanceDef::Virtual(def_id, _) = instance.def {
        let upcast_ty = match tcx.trait_of_item(def_id) {
            Some(trait_id) => trait_object_ty(
                tcx,
                ty::Binder::dummy(ty::TraitRef::from_method(tcx, trait_id, instance.args)),
            ),
            // drop_in_place won't have a defining trait, skip the upcast
            None => instance.args.type_at(0),
        };
        let stripped_ty = strip_receiver_auto(tcx, upcast_ty);
        instance.args = tcx.mk_args_trait(stripped_ty, instance.args.into_iter().skip(1));
    } else if let ty::InstanceDef::VTableShim(def_id) = instance.def
        && let Some(trait_id) = tcx.trait_of_item(def_id)
    {
        // VTableShims may have a trait method, but a concrete Self. This is not suitable for a vtable,
        // as the caller will not know the concrete Self.
        let trait_ref = ty::TraitRef::new(tcx, trait_id, instance.args);
        let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
        instance.args = tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
    }

    if options.contains(typeid::Options::ERASE_SELF_TYPE) {
        if let Some(impl_id) = tcx.impl_of_method(instance.def_id())
            && let Some(trait_ref) = tcx.impl_trait_ref(impl_id)
        {
            let impl_method = tcx.associated_item(instance.def_id());
            let method_id = impl_method
                .trait_item_def_id
                .expect("Part of a trait implementation, but not linked to the def_id?");
            let trait_method = tcx.associated_item(method_id);
            let trait_id = trait_ref.skip_binder().def_id;
            if traits::is_vtable_safe_method(tcx, trait_id, trait_method)
                && tcx.object_safety_violations(trait_id).is_empty()
            {
                // Trait methods will have a Self polymorphic parameter, where the concreteized
                // implementatation will not. We need to walk back to the more general trait method
                let trait_ref = tcx.instantiate_and_normalize_erasing_regions(
                    instance.args,
                    ty::ParamEnv::reveal_all(),
                    trait_ref,
                );
                let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));

                // At the call site, any call to this concrete function through a vtable will be
                // `Virtual(method_id, idx)` with appropriate arguments for the method. Since we have the
                // original method id, and we've recovered the trait arguments, we can make the callee
                // instance we're computing the alias set for match the caller instance.
                //
                // Right now, our code ignores the vtable index everywhere, so we use 0 as a placeholder.
                // If we ever *do* start encoding the vtable index, we will need to generate an alias set
                // based on which vtables we are putting this method into, as there will be more than one
                // index value when supertraits are involved.
                instance.def = ty::InstanceDef::Virtual(method_id, 0);
                let abstract_trait_args =
                    tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
                instance.args = instance.args.rebase_onto(tcx, impl_id, abstract_trait_args);
            }
        } else if tcx.is_closure_like(instance.def_id()) {
            // We're either a closure or a coroutine. Our goal is to find the trait we're defined on,
            // instantiate it, and take the type of its only method as our own.
            let closure_ty = instance.ty(tcx, ty::ParamEnv::reveal_all());
            let (trait_id, inputs) = match closure_ty.kind() {
                ty::Closure(..) => {
                    let closure_args = instance.args.as_closure();
                    let trait_id = tcx.fn_trait_kind_to_def_id(closure_args.kind()).unwrap();
                    let tuple_args =
                        tcx.instantiate_bound_regions_with_erased(closure_args.sig()).inputs()[0];
                    (trait_id, Some(tuple_args))
                }
                ty::Coroutine(..) => match tcx.coroutine_kind(instance.def_id()).unwrap() {
                    hir::CoroutineKind::Coroutine(..) => (
                        tcx.require_lang_item(LangItem::Coroutine, None),
                        Some(instance.args.as_coroutine().resume_ty()),
                    ),
                    hir::CoroutineKind::Desugared(desugaring, _) => {
                        let lang_item = match desugaring {
                            hir::CoroutineDesugaring::Async => LangItem::Future,
                            hir::CoroutineDesugaring::AsyncGen => LangItem::AsyncIterator,
                            hir::CoroutineDesugaring::Gen => LangItem::Iterator,
                        };
                        (tcx.require_lang_item(lang_item, None), None)
                    }
                },
                ty::CoroutineClosure(..) => (
                    tcx.require_lang_item(LangItem::FnOnce, None),
                    Some(
                        tcx.instantiate_bound_regions_with_erased(
                            instance.args.as_coroutine_closure().coroutine_closure_sig(),
                        )
                        .tupled_inputs_ty,
                    ),
                ),
                x => bug!("Unexpected type kind for closure-like: {x:?}"),
            };
            let concrete_args = tcx.mk_args_trait(closure_ty, inputs.map(Into::into));
            let trait_ref = ty::TraitRef::new(tcx, trait_id, concrete_args);
            let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
            let abstract_args = tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
            // There should be exactly one method on this trait, and it should be the one we're
            // defining.
            let call = tcx
                .associated_items(trait_id)
                .in_definition_order()
                .find(|it| it.kind == ty::AssocKind::Fn)
                .expect("No call-family function on closure-like Fn trait?")
                .def_id;

            instance.def = ty::InstanceDef::Virtual(call, 0);
            instance.args = abstract_args;
        }
    }
    instance
}

fn strip_receiver_auto<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    let ty::Dynamic(preds, lifetime, kind) = ty.kind() else {
        bug!("Tried to strip auto traits from non-dynamic type {ty}");
    };
    if preds.principal().is_some() {
        let filtered_preds =
            tcx.mk_poly_existential_predicates_from_iter(preds.into_iter().filter(|pred| {
                !matches!(pred.skip_binder(), ty::ExistentialPredicate::AutoTrait(..))
            }));
        Ty::new_dynamic(tcx, filtered_preds, *lifetime, *kind)
    } else {
        // If there's no principal type, re-encode it as a unit, since we don't know anything
        // about it. This technically discards the knowledge that it was a type that was made
        // into a trait object at some point, but that's not a lot.
        tcx.types.unit
    }
}

#[instrument(skip(tcx), ret)]
fn trait_object_ty<'tcx>(tcx: TyCtxt<'tcx>, poly_trait_ref: ty::PolyTraitRef<'tcx>) -> Ty<'tcx> {
    assert!(!poly_trait_ref.has_non_region_param());
    let principal_pred = poly_trait_ref.map_bound(|trait_ref| {
        ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref))
    });
    let mut assoc_preds: Vec<_> = traits::supertraits(tcx, poly_trait_ref)
        .flat_map(|super_poly_trait_ref| {
            tcx.associated_items(super_poly_trait_ref.def_id())
                .in_definition_order()
                .filter(|item| item.kind == ty::AssocKind::Type)
                .map(move |assoc_ty| {
                    super_poly_trait_ref.map_bound(|super_trait_ref| {
                        let alias_ty = ty::AliasTy::new(tcx, assoc_ty.def_id, super_trait_ref.args);
                        let resolved = tcx.normalize_erasing_regions(
                            ty::ParamEnv::reveal_all(),
                            alias_ty.to_ty(tcx),
                        );
                        debug!("Resolved {:?} -> {resolved}", alias_ty.to_ty(tcx));
                        ty::ExistentialPredicate::Projection(ty::ExistentialProjection {
                            def_id: assoc_ty.def_id,
                            args: ty::ExistentialTraitRef::erase_self_ty(tcx, super_trait_ref).args,
                            term: resolved.into(),
                        })
                    })
                })
        })
        .collect();
    assoc_preds.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
    let preds = tcx.mk_poly_existential_predicates_from_iter(
        iter::once(principal_pred).chain(assoc_preds.into_iter()),
    );
    Ty::new_dynamic(tcx, preds, tcx.lifetimes.re_erased, ty::Dyn)
}
