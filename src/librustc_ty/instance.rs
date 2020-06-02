use rustc_errors::ErrorReported;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Instance, TyCtxt, TypeFoldable};
use rustc_span::sym;
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;
use traits::{translate_substs, Reveal};

use log::debug;

fn resolve_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, (DefId, SubstsRef<'tcx>)>,
) -> Result<Option<Instance<'tcx>>, ErrorReported> {
    let (param_env, (def_id, substs)) = key.into_parts();

    debug!("resolve(def_id={:?}, substs={:?})", def_id, substs);
    let result = if let Some(trait_def_id) = tcx.trait_of_item(def_id) {
        debug!(" => associated item, attempting to find impl in param_env {:#?}", param_env);
        let item = tcx.associated_item(def_id);
        resolve_associated_item(tcx, &item, param_env, trait_def_id, substs)
    } else {
        let ty = tcx.type_of(def_id);
        let item_type = tcx.subst_and_normalize_erasing_regions(substs, param_env, &ty);

        let def = match item_type.kind {
            ty::FnDef(..)
                if {
                    let f = item_type.fn_sig(tcx);
                    f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic
                } =>
            {
                debug!(" => intrinsic");
                ty::InstanceDef::Intrinsic(def_id)
            }
            ty::FnDef(def_id, substs) if Some(def_id) == tcx.lang_items().drop_in_place_fn() => {
                let ty = substs.type_at(0);

                if ty.needs_drop(tcx, param_env) {
                    // `DropGlue` requires a monomorphic aka concrete type.
                    if ty.needs_subst() {
                        return Ok(None);
                    }

                    debug!(" => nontrivial drop glue");
                    ty::InstanceDef::DropGlue(def_id, Some(ty))
                } else {
                    debug!(" => trivial drop glue");
                    ty::InstanceDef::DropGlue(def_id, None)
                }
            }
            _ => {
                debug!(" => free item");
                ty::InstanceDef::Item(def_id)
            }
        };
        Ok(Some(Instance { def, substs }))
    };
    debug!("resolve(def_id={:?}, substs={:?}) = {:?}", def_id, substs, result);
    result
}

fn resolve_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item: &ty::AssocItem,
    param_env: ty::ParamEnv<'tcx>,
    trait_id: DefId,
    rcvr_substs: SubstsRef<'tcx>,
) -> Result<Option<Instance<'tcx>>, ErrorReported> {
    let def_id = trait_item.def_id;
    debug!(
        "resolve_associated_item(trait_item={:?}, \
            param_env={:?}, \
            trait_id={:?}, \
            rcvr_substs={:?})",
        def_id, param_env, trait_id, rcvr_substs
    );

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = tcx.codegen_fulfill_obligation((param_env, ty::Binder::bind(trait_ref)))?;

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    Ok(match vtbl {
        traits::ImplSourceUserDefined(impl_data) => {
            debug!(
                "resolving ImplSourceUserDefined: {:?}, {:?}, {:?}, {:?}",
                param_env, trait_item, rcvr_substs, impl_data
            );
            assert!(!rcvr_substs.needs_infer());
            assert!(!trait_ref.needs_infer());

            let trait_def_id = tcx.trait_id_of_impl(impl_data.impl_def_id).unwrap();
            let trait_def = tcx.trait_def(trait_def_id);
            let leaf_def = trait_def
                .ancestors(tcx, impl_data.impl_def_id)?
                .leaf_def(tcx, trait_item.ident, trait_item.kind)
                .unwrap_or_else(|| {
                    bug!("{:?} not found in {:?}", trait_item, impl_data.impl_def_id);
                });

            let substs = tcx.infer_ctxt().enter(|infcx| {
                let param_env = param_env.with_reveal_all();
                let substs = rcvr_substs.rebase_onto(tcx, trait_def_id, impl_data.substs);
                let substs = translate_substs(
                    &infcx,
                    param_env,
                    impl_data.impl_def_id,
                    substs,
                    leaf_def.defining_node,
                );
                infcx.tcx.erase_regions(&substs)
            });

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
                if param_env.reveal == Reveal::All {
                    !trait_ref.still_further_specializable()
                } else {
                    false
                }
            };

            if !eligible {
                return Ok(None);
            }

            let substs = tcx.erase_regions(&substs);

            // Check if we just resolved an associated `const` declaration from
            // a `trait` to an associated `const` definition in an `impl`, where
            // the definition in the `impl` has the wrong type (for which an
            // error has already been/will be emitted elsewhere).
            //
            // NB: this may be expensive, we try to skip it in all the cases where
            // we know the error would've been caught (e.g. in an upstream crate).
            //
            // A better approach might be to just introduce a query (returning
            // `Result<(), ErrorReported>`) for the check that `rustc_typeck`
            // performs (i.e. that the definition's type in the `impl` matches
            // the declaration in the `trait`), so that we can cheaply check
            // here if it failed, instead of approximating it.
            if trait_item.kind == ty::AssocKind::Const
                && trait_item.def_id != leaf_def.item.def_id
                && leaf_def.item.def_id.is_local()
            {
                let normalized_type_of = |def_id, substs| {
                    tcx.subst_and_normalize_erasing_regions(substs, param_env, &tcx.type_of(def_id))
                };

                let original_ty = normalized_type_of(trait_item.def_id, rcvr_substs);
                let resolved_ty = normalized_type_of(leaf_def.item.def_id, substs);

                if original_ty != resolved_ty {
                    let msg = format!(
                        "Instance::resolve: inconsistent associated `const` type: \
                         was `{}: {}` but resolved to `{}: {}`",
                        tcx.def_path_str_with_substs(trait_item.def_id, rcvr_substs),
                        original_ty,
                        tcx.def_path_str_with_substs(leaf_def.item.def_id, substs),
                        resolved_ty,
                    );
                    let span = tcx.def_span(leaf_def.item.def_id);
                    tcx.sess.delay_span_bug(span, &msg);

                    return Err(ErrorReported);
                }
            }

            Some(ty::Instance::new(leaf_def.item.def_id, substs))
        }
        traits::ImplSourceGenerator(generator_data) => Some(Instance {
            def: ty::InstanceDef::Item(generator_data.generator_def_id),
            substs: generator_data.substs,
        }),
        traits::ImplSourceClosure(closure_data) => {
            let trait_closure_kind = tcx.fn_trait_kind_from_lang_item(trait_id).unwrap();
            Some(Instance::resolve_closure(
                tcx,
                closure_data.closure_def_id,
                closure_data.substs,
                trait_closure_kind,
            ))
        }
        traits::ImplSourceFnPointer(ref data) => {
            // `FnPtrShim` requires a monomorphic aka concrete type.
            if data.fn_ty.needs_subst() {
                return Ok(None);
            }

            Some(Instance {
                def: ty::InstanceDef::FnPtrShim(trait_item.def_id, data.fn_ty),
                substs: rcvr_substs,
            })
        }
        traits::ImplSourceObject(ref data) => {
            let index = traits::get_vtable_index_of_object_method(tcx, data, def_id);
            Some(Instance { def: ty::InstanceDef::Virtual(def_id, index), substs: rcvr_substs })
        }
        traits::ImplSourceBuiltin(..) => {
            if Some(trait_ref.def_id) == tcx.lang_items().clone_trait() {
                // FIXME(eddyb) use lang items for methods instead of names.
                let name = tcx.item_name(def_id);
                if name == sym::clone {
                    let self_ty = trait_ref.self_ty();

                    // `CloneShim` requires a monomorphic aka concrete type.
                    if self_ty.needs_subst() {
                        return Ok(None);
                    }

                    Some(Instance {
                        def: ty::InstanceDef::CloneShim(def_id, self_ty),
                        substs: rcvr_substs,
                    })
                } else {
                    assert_eq!(name, sym::clone_from);

                    // Use the default `fn clone_from` from `trait Clone`.
                    let substs = tcx.erase_regions(&rcvr_substs);
                    Some(ty::Instance::new(def_id, substs))
                }
            } else {
                None
            }
        }
        traits::ImplSourceAutoImpl(..)
        | traits::ImplSourceParam(..)
        | traits::ImplSourceTraitAlias(..)
        | traits::ImplSourceDiscriminantKind(..) => None,
    })
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    *providers = ty::query::Providers { resolve_instance, ..*providers };
}
