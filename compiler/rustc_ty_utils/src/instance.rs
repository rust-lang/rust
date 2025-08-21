use rustc_errors::ErrorGuaranteed;
use rustc_hir::LangItem;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::traits::{BuiltinImplSource, CodegenObligationError};
use rustc_middle::ty::{
    self, ClosureKind, GenericArgsRef, Instance, PseudoCanonicalInput, TyCtxt, TypeVisitableExt,
};
use rustc_span::sym;
use rustc_trait_selection::traits;
use tracing::debug;
use traits::translate_args;

use crate::errors::UnexpectedFnPtrAssociatedItem;

fn resolve_instance_raw<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, (DefId, GenericArgsRef<'tcx>)>,
) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
    let PseudoCanonicalInput { typing_env, value: (def_id, args) } = key;

    let result = if let Some(trait_def_id) = tcx.trait_of_assoc(def_id) {
        debug!(" => associated item, attempting to find impl in typing_env {:#?}", typing_env);
        resolve_associated_item(
            tcx,
            def_id,
            typing_env,
            trait_def_id,
            tcx.normalize_erasing_regions(typing_env, args),
        )
    } else {
        let def = if tcx.intrinsic(def_id).is_some() {
            debug!(" => intrinsic");
            ty::InstanceKind::Intrinsic(def_id)
        } else if tcx.is_lang_item(def_id, LangItem::DropInPlace) {
            let ty = args.type_at(0);

            if ty.needs_drop(tcx, typing_env) {
                debug!(" => nontrivial drop glue");
                match *ty.kind() {
                    ty::Coroutine(coroutine_def_id, ..) => {
                        // FIXME: sync drop of coroutine with async drop (generate both versions?)
                        // Currently just ignored
                        if tcx.optimized_mir(coroutine_def_id).coroutine_drop_async().is_some() {
                            ty::InstanceKind::DropGlue(def_id, None)
                        } else {
                            ty::InstanceKind::DropGlue(def_id, Some(ty))
                        }
                    }
                    ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Tuple(..)
                    | ty::Adt(..)
                    | ty::Dynamic(..)
                    | ty::Array(..)
                    | ty::Slice(..)
                    | ty::UnsafeBinder(..) => ty::InstanceKind::DropGlue(def_id, Some(ty)),
                    // Drop shims can only be built from ADTs.
                    _ => return Ok(None),
                }
            } else {
                debug!(" => trivial drop glue");
                ty::InstanceKind::DropGlue(def_id, None)
            }
        } else if tcx.is_lang_item(def_id, LangItem::AsyncDropInPlace) {
            let ty = args.type_at(0);

            if ty.needs_async_drop(tcx, typing_env) {
                match *ty.kind() {
                    ty::Closure(..)
                    | ty::CoroutineClosure(..)
                    | ty::Coroutine(..)
                    | ty::Tuple(..)
                    | ty::Adt(..)
                    | ty::Dynamic(..)
                    | ty::Array(..)
                    | ty::Slice(..) => {}
                    // Async destructor ctor shims can only be built from ADTs.
                    _ => return Ok(None),
                }
                debug!(" => nontrivial async drop glue ctor");
                ty::InstanceKind::AsyncDropGlueCtorShim(def_id, ty)
            } else {
                debug!(" => trivial async drop glue ctor");
                ty::InstanceKind::AsyncDropGlueCtorShim(def_id, ty)
            }
        } else if tcx.is_async_drop_in_place_coroutine(def_id) {
            let ty = args.type_at(0);
            ty::InstanceKind::AsyncDropGlue(def_id, ty)
        } else {
            debug!(" => free item");
            ty::InstanceKind::Item(def_id)
        };

        Ok(Some(Instance { def, args }))
    };
    debug!("resolve_instance: result={:?}", result);
    result
}

fn resolve_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item_id: DefId,
    typing_env: ty::TypingEnv<'tcx>,
    trait_id: DefId,
    rcvr_args: GenericArgsRef<'tcx>,
) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
    debug!(?trait_item_id, ?typing_env, ?trait_id, ?rcvr_args, "resolve_associated_item");

    let trait_ref = ty::TraitRef::from_assoc(tcx, trait_id, rcvr_args);

    let input = typing_env.as_query_input(trait_ref);
    let vtbl = match tcx.codegen_select_candidate(input) {
        Ok(vtbl) => vtbl,
        Err(CodegenObligationError::Ambiguity | CodegenObligationError::Unimplemented) => {
            return Ok(None);
        }
        Err(CodegenObligationError::UnconstrainedParam(guar)) => return Err(guar),
    };

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    Ok(match vtbl {
        traits::ImplSource::UserDefined(impl_data) => {
            debug!(
                "resolving ImplSource::UserDefined: {:?}, {:?}, {:?}, {:?}",
                typing_env, trait_item_id, rcvr_args, impl_data
            );
            assert!(!rcvr_args.has_infer());
            assert!(!trait_ref.has_infer());

            let trait_def_id = tcx.trait_id_of_impl(impl_data.impl_def_id).unwrap();
            let trait_def = tcx.trait_def(trait_def_id);
            let leaf_def = trait_def
                .ancestors(tcx, impl_data.impl_def_id)?
                .leaf_def(tcx, trait_item_id)
                .unwrap_or_else(|| {
                    bug!("{:?} not found in {:?}", trait_item_id, impl_data.impl_def_id);
                });

            // Since this is a trait item, we need to see if the item is either a trait
            // default item or a specialization because we can't resolve those until we're
            // in `TypingMode::PostAnalysis`.
            //
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
                match typing_env.typing_mode {
                    ty::TypingMode::Coherence
                    | ty::TypingMode::Analysis { .. }
                    | ty::TypingMode::Borrowck { .. }
                    | ty::TypingMode::PostBorrowckAnalysis { .. } => false,
                    ty::TypingMode::PostAnalysis => !trait_ref.still_further_specializable(),
                }
            };
            if !eligible {
                return Ok(None);
            }

            let typing_env = typing_env.with_post_analysis_normalized(tcx);
            let (infcx, param_env) = tcx.infer_ctxt().build_with_typing_env(typing_env);
            let args = rcvr_args.rebase_onto(tcx, trait_def_id, impl_data.args);
            let args = translate_args(
                &infcx,
                param_env,
                impl_data.impl_def_id,
                args,
                leaf_def.defining_node,
            );
            let args = infcx.tcx.erase_and_anonymize_regions(args);

            // HACK: We may have overlapping `dyn Trait` built-in impls and
            // user-provided blanket impls. Detect that case here, and return
            // ambiguity.
            //
            // This should not affect totally monomorphized contexts, only
            // resolve calls that happen polymorphically, such as the mir-inliner
            // and const-prop (and also some lints).
            let self_ty = rcvr_args.type_at(0);
            if !self_ty.is_known_rigid() {
                let predicates = tcx
                    .predicates_of(impl_data.impl_def_id)
                    .instantiate(tcx, impl_data.args)
                    .predicates;
                let sized_def_id = tcx.lang_items().sized_trait();
                // If we find a `Self: Sized` bound on the item, then we know
                // that `dyn Trait` can certainly never apply here.
                if !predicates.into_iter().filter_map(ty::Clause::as_trait_clause).any(|clause| {
                    Some(clause.def_id()) == sized_def_id
                        && clause.skip_binder().self_ty() == self_ty
                }) {
                    return Ok(None);
                }
            }

            // Any final impl is required to define all associated items.
            if !leaf_def.item.defaultness(tcx).has_value() {
                let guar = tcx.dcx().span_delayed_bug(
                    tcx.def_span(leaf_def.item.def_id),
                    "missing value for assoc item in impl",
                );
                return Err(guar);
            }

            // Make sure that we're projecting to an item that has compatible args.
            // This may happen if we are resolving an instance before codegen, such
            // as during inlining. This check is also done in projection.
            if !tcx.check_args_compatible(leaf_def.item.def_id, args) {
                let guar = tcx.dcx().span_delayed_bug(
                    tcx.def_span(leaf_def.item.def_id),
                    "missing value for assoc item in impl",
                );
                return Err(guar);
            }

            let args = tcx.erase_and_anonymize_regions(args);

            // We check that the impl item is compatible with the trait item
            // because otherwise we may ICE in const eval due to type mismatches,
            // signature incompatibilities, etc.
            // NOTE: We could also only enforce this in `PostAnalysis`, which
            // is what CTFE and MIR inlining would care about anyways.
            if trait_item_id != leaf_def.item.def_id
                && let Some(leaf_def_item) = leaf_def.item.def_id.as_local()
            {
                tcx.ensure_ok().compare_impl_item(leaf_def_item)?;
            }

            Some(ty::Instance::new_raw(leaf_def.item.def_id, args))
        }
        traits::ImplSource::Builtin(BuiltinImplSource::Object(_), _) => {
            let trait_ref = ty::TraitRef::from_assoc(tcx, trait_id, rcvr_args);
            if trait_ref.has_non_region_infer() || trait_ref.has_non_region_param() {
                // We only resolve totally substituted vtable entries.
                None
            } else {
                let vtable_base = tcx.first_method_vtable_slot(trait_ref);
                let offset = tcx
                    .own_existential_vtable_entries(trait_id)
                    .iter()
                    .copied()
                    .position(|def_id| def_id == trait_item_id);
                offset.map(|offset| Instance {
                    def: ty::InstanceKind::Virtual(trait_item_id, vtable_base + offset),
                    args: rcvr_args,
                })
            }
        }
        traits::ImplSource::Builtin(BuiltinImplSource::Misc | BuiltinImplSource::Trivial, _) => {
            if tcx.is_lang_item(trait_ref.def_id, LangItem::Clone) {
                // FIXME(eddyb) use lang items for methods instead of names.
                let name = tcx.item_name(trait_item_id);
                if name == sym::clone {
                    let self_ty = trait_ref.self_ty();
                    match self_ty.kind() {
                        ty::FnDef(..) | ty::FnPtr(..) => (),
                        ty::Coroutine(..)
                        | ty::CoroutineWitness(..)
                        | ty::Closure(..)
                        | ty::CoroutineClosure(..)
                        | ty::Tuple(..) => {}
                        _ => return Ok(None),
                    };

                    Some(Instance {
                        def: ty::InstanceKind::CloneShim(trait_item_id, self_ty),
                        args: rcvr_args,
                    })
                } else {
                    assert_eq!(name, sym::clone_from);

                    // Use the default `fn clone_from` from `trait Clone`.
                    let args = tcx.erase_and_anonymize_regions(rcvr_args);
                    Some(ty::Instance::new_raw(trait_item_id, args))
                }
            } else if tcx.is_lang_item(trait_ref.def_id, LangItem::FnPtrTrait) {
                if tcx.is_lang_item(trait_item_id, LangItem::FnPtrAddr) {
                    let self_ty = trait_ref.self_ty();
                    if !matches!(self_ty.kind(), ty::FnPtr(..)) {
                        return Ok(None);
                    }
                    Some(Instance {
                        def: ty::InstanceKind::FnPtrAddrShim(trait_item_id, self_ty),
                        args: rcvr_args,
                    })
                } else {
                    tcx.dcx().emit_fatal(UnexpectedFnPtrAssociatedItem {
                        span: tcx.def_span(trait_item_id),
                    })
                }
            } else if let Some(target_kind) = tcx.fn_trait_kind_from_def_id(trait_ref.def_id) {
                // FIXME: This doesn't check for malformed libcore that defines, e.g.,
                // `trait Fn { fn call_once(&self) { .. } }`. This is mostly for extension
                // methods.
                if cfg!(debug_assertions)
                    && ![sym::call, sym::call_mut, sym::call_once]
                        .contains(&tcx.item_name(trait_item_id))
                {
                    // For compiler developers who'd like to add new items to `Fn`/`FnMut`/`FnOnce`,
                    // you either need to generate a shim body, or perhaps return
                    // `InstanceKind::Item` pointing to a trait default method body if
                    // it is given a default implementation by the trait.
                    bug!(
                        "no definition for `{trait_ref}::{}` for built-in callable type",
                        tcx.item_name(trait_item_id)
                    )
                }
                match *rcvr_args.type_at(0).kind() {
                    ty::Closure(closure_def_id, args) => {
                        Some(Instance::resolve_closure(tcx, closure_def_id, args, target_kind))
                    }
                    ty::FnDef(..) | ty::FnPtr(..) => Some(Instance {
                        def: ty::InstanceKind::FnPtrShim(trait_item_id, rcvr_args.type_at(0)),
                        args: rcvr_args,
                    }),
                    ty::CoroutineClosure(coroutine_closure_def_id, args) => {
                        // When a coroutine-closure implements the `Fn` traits, then it
                        // always dispatches to the `FnOnce` implementation. This is to
                        // ensure that the `closure_kind` of the resulting closure is in
                        // sync with the built-in trait implementations (since all of the
                        // implementations return `FnOnce::Output`).
                        if ty::ClosureKind::FnOnce == args.as_coroutine_closure().kind() {
                            Some(Instance::new_raw(coroutine_closure_def_id, args))
                        } else {
                            Some(Instance {
                                def: ty::InstanceKind::ConstructCoroutineInClosureShim {
                                    coroutine_closure_def_id,
                                    receiver_by_ref: target_kind != ty::ClosureKind::FnOnce,
                                },
                                args,
                            })
                        }
                    }
                    _ => bug!(
                        "no built-in definition for `{trait_ref}::{}` for non-fn type",
                        tcx.item_name(trait_item_id)
                    ),
                }
            } else if let Some(target_kind) = tcx.async_fn_trait_kind_from_def_id(trait_ref.def_id)
            {
                match *rcvr_args.type_at(0).kind() {
                    ty::CoroutineClosure(coroutine_closure_def_id, args) => {
                        if target_kind == ClosureKind::FnOnce
                            && args.as_coroutine_closure().kind() != ClosureKind::FnOnce
                        {
                            // If we're computing `AsyncFnOnce` for a by-ref closure then
                            // construct a new body that has the right return types.
                            Some(Instance {
                                def: ty::InstanceKind::ConstructCoroutineInClosureShim {
                                    coroutine_closure_def_id,
                                    receiver_by_ref: false,
                                },
                                args,
                            })
                        } else {
                            Some(Instance::new_raw(coroutine_closure_def_id, args))
                        }
                    }
                    ty::Closure(closure_def_id, args) => {
                        Some(Instance::resolve_closure(tcx, closure_def_id, args, target_kind))
                    }
                    ty::FnDef(..) | ty::FnPtr(..) => Some(Instance {
                        def: ty::InstanceKind::FnPtrShim(trait_item_id, rcvr_args.type_at(0)),
                        args: rcvr_args,
                    }),
                    _ => bug!(
                        "no built-in definition for `{trait_ref}::{}` for non-lending-closure type",
                        tcx.item_name(trait_item_id)
                    ),
                }
            } else if tcx.is_lang_item(trait_ref.def_id, LangItem::TransmuteTrait) {
                let name = tcx.item_name(trait_item_id);
                assert_eq!(name, sym::transmute);
                let args = tcx.erase_and_anonymize_regions(rcvr_args);
                Some(ty::Instance::new_raw(trait_item_id, args))
            } else {
                Instance::try_resolve_item_for_coroutine(tcx, trait_item_id, trait_id, rcvr_args)
            }
        }
        traits::ImplSource::Param(..)
        | traits::ImplSource::Builtin(BuiltinImplSource::TraitUpcasting { .. }, _) => None,
    })
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { resolve_instance_raw, ..*providers };
}
