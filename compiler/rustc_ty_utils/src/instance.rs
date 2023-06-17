use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::query::Providers;
use rustc_middle::traits::CodegenObligationError;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Instance, TyCtxt, TypeVisitableExt};
use rustc_span::sym;
use rustc_trait_selection::traits;
use traits::{translate_substs, Reveal};

use crate::errors::UnexpectedFnPtrAssociatedItem;

fn resolve_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, (DefId, SubstsRef<'tcx>)>,
) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
    let (param_env, (def, substs)) = key.into_parts();

    let result = if let Some(trait_def_id) = tcx.trait_of_item(def) {
        debug!(" => associated item, attempting to find impl in param_env {:#?}", param_env);
        resolve_associated_item(
            tcx,
            def,
            param_env,
            trait_def_id,
            tcx.normalize_erasing_regions(param_env, substs),
        )
    } else {
        let ty = tcx.type_of(def);
        let item_type = tcx.subst_and_normalize_erasing_regions(substs, param_env, ty);

        let def = match *item_type.kind() {
            ty::FnDef(def_id, ..) if tcx.is_intrinsic(def_id) => {
                debug!(" => intrinsic");
                ty::InstanceDef::Intrinsic(def)
            }
            ty::FnDef(def_id, substs) if Some(def_id) == tcx.lang_items().drop_in_place_fn() => {
                let ty = substs.type_at(0);

                if ty.needs_drop(tcx, param_env) {
                    debug!(" => nontrivial drop glue");
                    match *ty.kind() {
                        ty::Closure(..)
                        | ty::Generator(..)
                        | ty::Tuple(..)
                        | ty::Adt(..)
                        | ty::Dynamic(..)
                        | ty::Array(..)
                        | ty::Slice(..) => {}
                        // Drop shims can only be built from ADTs.
                        _ => return Ok(None),
                    }

                    ty::InstanceDef::DropGlue(def_id, Some(ty))
                } else {
                    debug!(" => trivial drop glue");
                    ty::InstanceDef::DropGlue(def_id, None)
                }
            }
            _ => {
                debug!(" => free item");
                ty::InstanceDef::Item(def)
            }
        };
        Ok(Some(Instance { def, substs }))
    };
    debug!("inner_resolve_instance: result={:?}", result);
    result
}

fn resolve_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item_id: DefId,
    param_env: ty::ParamEnv<'tcx>,
    trait_id: DefId,
    rcvr_substs: SubstsRef<'tcx>,
) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
    debug!(?trait_item_id, ?param_env, ?trait_id, ?rcvr_substs, "resolve_associated_item");

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);

    let vtbl = match tcx.codegen_select_candidate((param_env, ty::Binder::dummy(trait_ref))) {
        Ok(vtbl) => vtbl,
        Err(CodegenObligationError::Ambiguity) => {
            let reported = tcx.sess.delay_span_bug(
                tcx.def_span(trait_item_id),
                format!(
                    "encountered ambiguity selecting `{trait_ref:?}` during codegen, presuming due to \
                     overflow or prior type error",
                ),
            );
            return Err(reported);
        }
        Err(CodegenObligationError::Unimplemented) => return Ok(None),
        Err(CodegenObligationError::FulfillmentError) => return Ok(None),
    };

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    Ok(match vtbl {
        traits::ImplSource::UserDefined(impl_data) => {
            debug!(
                "resolving ImplSource::UserDefined: {:?}, {:?}, {:?}, {:?}",
                param_env, trait_item_id, rcvr_substs, impl_data
            );
            assert!(!rcvr_substs.has_infer());
            assert!(!trait_ref.has_infer());

            let trait_def_id = tcx.trait_id_of_impl(impl_data.impl_def_id).unwrap();
            let trait_def = tcx.trait_def(trait_def_id);
            let leaf_def = trait_def
                .ancestors(tcx, impl_data.impl_def_id)?
                .leaf_def(tcx, trait_item_id)
                .unwrap_or_else(|| {
                    bug!("{:?} not found in {:?}", trait_item_id, impl_data.impl_def_id);
                });
            let infcx = tcx.infer_ctxt().build();
            let param_env = param_env.with_reveal_all_normalized(tcx);
            let substs = rcvr_substs.rebase_onto(tcx, trait_def_id, impl_data.substs);
            let substs = translate_substs(
                &infcx,
                param_env,
                impl_data.impl_def_id,
                substs,
                leaf_def.defining_node,
            );
            let substs = infcx.tcx.erase_regions(substs);

            // Since this is a trait item, we need to see if the item is either a trait default item
            // or a specialization because we can't resolve those unless we can `Reveal::All`.
            // NOTE: This should be kept in sync with the similar code in
            // `rustc_trait_selection::traits::project::assemble_candidates_from_impls()`.
            let eligible = if leaf_def.is_final() {
                // Non-specializable items are always projectable.
                true
            } else {
                // Only reveal a specializable default if we're past type-checking
                // and the obligation is monomorphic, otherwise passes such as
                // transmute checking and polymorphic MIR optimizations could
                // get a result which isn't correct for all monomorphizations.
                if param_env.reveal() == Reveal::All {
                    !trait_ref.still_further_specializable()
                } else {
                    false
                }
            };

            if !eligible {
                return Ok(None);
            }

            // Any final impl is required to define all associated items.
            if !leaf_def.item.defaultness(tcx).has_value() {
                let guard = tcx.sess.delay_span_bug(
                    tcx.def_span(leaf_def.item.def_id),
                    "missing value for assoc item in impl",
                );
                return Err(guard);
            }

            let substs = tcx.erase_regions(substs);

            // Check if we just resolved an associated `const` declaration from
            // a `trait` to an associated `const` definition in an `impl`, where
            // the definition in the `impl` has the wrong type (for which an
            // error has already been/will be emitted elsewhere).
            if leaf_def.item.kind == ty::AssocKind::Const
                && trait_item_id != leaf_def.item.def_id
                && let Some(leaf_def_item) = leaf_def.item.def_id.as_local()
            {
                tcx.compare_impl_const((
                    leaf_def_item,
                    trait_item_id,
                ))?;
            }

            Some(ty::Instance::new(leaf_def.item.def_id, substs))
        }
        traits::ImplSource::Object(ref data) => {
            traits::get_vtable_index_of_object_method(tcx, data, trait_item_id).map(|index| {
                Instance {
                    def: ty::InstanceDef::Virtual(trait_item_id, index),
                    substs: rcvr_substs,
                }
            })
        }
        traits::ImplSource::Builtin(..) => {
            let lang_items = tcx.lang_items();
            if Some(trait_ref.def_id) == lang_items.clone_trait() {
                // FIXME(eddyb) use lang items for methods instead of names.
                let name = tcx.item_name(trait_item_id);
                if name == sym::clone {
                    let self_ty = trait_ref.self_ty();

                    let is_copy = self_ty.is_copy_modulo_regions(tcx, param_env);
                    match self_ty.kind() {
                        _ if is_copy => (),
                        ty::Generator(..)
                        | ty::GeneratorWitness(..)
                        | ty::Closure(..)
                        | ty::Tuple(..) => {}
                        _ => return Ok(None),
                    };

                    Some(Instance {
                        def: ty::InstanceDef::CloneShim(trait_item_id, self_ty),
                        substs: rcvr_substs,
                    })
                } else {
                    assert_eq!(name, sym::clone_from);

                    // Use the default `fn clone_from` from `trait Clone`.
                    let substs = tcx.erase_regions(rcvr_substs);
                    Some(ty::Instance::new(trait_item_id, substs))
                }
            } else if Some(trait_ref.def_id) == lang_items.fn_ptr_trait() {
                if lang_items.fn_ptr_addr() == Some(trait_item_id) {
                    let self_ty = trait_ref.self_ty();
                    if !matches!(self_ty.kind(), ty::FnPtr(..)) {
                        return Ok(None);
                    }
                    Some(Instance {
                        def: ty::InstanceDef::FnPtrAddrShim(trait_item_id, self_ty),
                        substs: rcvr_substs,
                    })
                } else {
                    tcx.sess.emit_fatal(UnexpectedFnPtrAssociatedItem {
                        span: tcx.def_span(trait_item_id),
                    })
                }
            } else if Some(trait_ref.def_id) == lang_items.future_trait() {
                let ty::Generator(generator_def_id, substs, _) = *rcvr_substs.type_at(0).kind() else {
                    bug!()
                };
                if Some(trait_item_id) == tcx.lang_items().future_poll_fn() {
                    // `Future::poll` is generated by the compiler.
                    Some(Instance { def: ty::InstanceDef::Item(generator_def_id), substs: substs })
                } else {
                    // All other methods are default methods of the `Future` trait.
                    // (this assumes that `ImplSource::Builtin` is only used for methods on `Future`)
                    debug_assert!(tcx.defaultness(trait_item_id).has_value());
                    Some(Instance::new(trait_item_id, rcvr_substs))
                }
            } else if Some(trait_ref.def_id) == lang_items.gen_trait() {
                let ty::Generator(generator_def_id, substs, _) = *rcvr_substs.type_at(0).kind() else {
                    bug!()
                };
                if cfg!(debug_assertions) && tcx.item_name(trait_item_id) != sym::resume {
                    // For compiler developers who'd like to add new items to `Generator`,
                    // you either need to generate a shim body, or perhaps return
                    // `InstanceDef::Item` pointing to a trait default method body if
                    // it is given a default implementation by the trait.
                    span_bug!(
                        tcx.def_span(generator_def_id),
                        "no definition for `{trait_ref}::{}` for built-in generator type",
                        tcx.item_name(trait_item_id)
                    )
                }
                Some(Instance { def: ty::InstanceDef::Item(generator_def_id), substs })
            } else if tcx.fn_trait_kind_from_def_id(trait_ref.def_id).is_some() {
                // FIXME: This doesn't check for malformed libcore that defines, e.g.,
                // `trait Fn { fn call_once(&self) { .. } }`. This is mostly for extension
                // methods.
                if cfg!(debug_assertions)
                    && ![sym::call, sym::call_mut, sym::call_once]
                        .contains(&tcx.item_name(trait_item_id))
                {
                    // For compiler developers who'd like to add new items to `Fn`/`FnMut`/`FnOnce`,
                    // you either need to generate a shim body, or perhaps return
                    // `InstanceDef::Item` pointing to a trait default method body if
                    // it is given a default implementation by the trait.
                    bug!(
                        "no definition for `{trait_ref}::{}` for built-in callable type",
                        tcx.item_name(trait_item_id)
                    )
                }
                match *rcvr_substs.type_at(0).kind() {
                    ty::Closure(closure_def_id, substs) => {
                        let trait_closure_kind = tcx.fn_trait_kind_from_def_id(trait_id).unwrap();
                        Instance::resolve_closure(tcx, closure_def_id, substs, trait_closure_kind)
                    }
                    ty::FnDef(..) | ty::FnPtr(..) => Some(Instance {
                        def: ty::InstanceDef::FnPtrShim(trait_item_id, rcvr_substs.type_at(0)),
                        substs: rcvr_substs,
                    }),
                    _ => bug!(
                        "no built-in definition for `{trait_ref}::{}` for non-fn type",
                        tcx.item_name(trait_item_id)
                    ),
                }
            } else {
                None
            }
        }
        traits::ImplSource::Param(..)
        | traits::ImplSource::TraitAlias(..)
        | traits::ImplSource::TraitUpcasting(_) => None,
    })
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { resolve_instance, ..*providers };
}
