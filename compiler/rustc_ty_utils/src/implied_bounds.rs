use std::iter;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, Ty, TyCtxt, fold_regions};
use rustc_span::Span;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        assumed_wf_types,
        assumed_wf_types_for_rpitit: |tcx, def_id| {
            assert!(tcx.is_impl_trait_in_trait(def_id.to_def_id()));
            tcx.assumed_wf_types(def_id)
        },
        ..*providers
    };
}

fn assumed_wf_types<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx [(Ty<'tcx>, Span)] {
    match tcx.def_kind(def_id) {
        DefKind::Fn => {
            let sig = tcx.fn_sig(def_id).instantiate_identity();
            let liberated_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), sig);
            tcx.arena.alloc_from_iter(itertools::zip_eq(
                liberated_sig.inputs_and_output,
                fn_sig_spans(tcx, def_id),
            ))
        }
        DefKind::AssocFn => {
            let sig = tcx.fn_sig(def_id).instantiate_identity();
            let liberated_sig = tcx.liberate_late_bound_regions(def_id.to_def_id(), sig);
            let mut assumed_wf_types: Vec<_> =
                tcx.assumed_wf_types(tcx.local_parent(def_id)).into();
            assumed_wf_types.extend(itertools::zip_eq(
                liberated_sig.inputs_and_output,
                fn_sig_spans(tcx, def_id),
            ));
            tcx.arena.alloc_slice(&assumed_wf_types)
        }
        DefKind::Impl { .. } => {
            // Trait arguments and the self type for trait impls or only the self type for
            // inherent impls.
            let tys = match tcx.impl_trait_ref(def_id) {
                Some(trait_ref) => trait_ref.skip_binder().args.types().collect(),
                None => vec![tcx.type_of(def_id).instantiate_identity()],
            };

            let mut impl_spans = impl_spans(tcx, def_id);
            tcx.arena.alloc_from_iter(tys.into_iter().map(|ty| (ty, impl_spans.next().unwrap())))
        }
        DefKind::AssocTy if let Some(data) = tcx.opt_rpitit_info(def_id.to_def_id()) => {
            match data {
                ty::ImplTraitInTraitData::Trait { fn_def_id, .. } => {
                    // We need to remap all of the late-bound lifetimes in the assumed wf types
                    // of the fn (which are represented as ReLateParam) to the early-bound lifetimes
                    // of the RPITIT (which are represented by ReEarlyParam owned by the opaque).
                    // Luckily, this is very easy to do because we already have that mapping
                    // stored in the HIR of this RPITIT.
                    //
                    // Side-note: We don't really need to do this remapping for early-bound
                    // lifetimes because they're already "linked" by the bidirectional outlives
                    // predicates we insert in the `explicit_predicates_of` query for RPITITs.
                    let mut mapping = FxHashMap::default();
                    let generics = tcx.generics_of(def_id);

                    // For each captured opaque lifetime, if it's late-bound (`ReLateParam` in this
                    // case, since it has been liberated), map it back to the early-bound lifetime of
                    // the GAT. Since RPITITs also have all of the fn's generics, we slice only
                    // the end of the list corresponding to the opaque's generics.
                    for param in &generics.own_params[tcx.generics_of(fn_def_id).own_params.len()..]
                    {
                        let orig_lt =
                            tcx.map_opaque_lifetime_to_parent_lifetime(param.def_id.expect_local());
                        if matches!(orig_lt.kind(), ty::ReLateParam(..)) {
                            mapping.insert(
                                orig_lt,
                                ty::Region::new_early_param(
                                    tcx,
                                    ty::EarlyParamRegion { index: param.index, name: param.name },
                                ),
                            );
                        }
                    }
                    // FIXME: This could use a real folder, I guess.
                    let remapped_wf_tys = fold_regions(
                        tcx,
                        tcx.assumed_wf_types(fn_def_id.expect_local()).to_vec(),
                        |region, _| {
                            // If `region` is a `ReLateParam` that is captured by the
                            // opaque, remap it to its corresponding the early-
                            // bound region.
                            if let Some(remapped_region) = mapping.get(&region) {
                                *remapped_region
                            } else {
                                region
                            }
                        },
                    );
                    tcx.arena.alloc_from_iter(remapped_wf_tys)
                }
                // Assumed wf types for RPITITs in an impl just inherit (and instantiate)
                // the assumed wf types of the trait's RPITIT GAT.
                ty::ImplTraitInTraitData::Impl { .. } => {
                    let impl_def_id = tcx.local_parent(def_id);
                    let rpitit_def_id = tcx.associated_item(def_id).trait_item_def_id.unwrap();
                    let args = ty::GenericArgs::identity_for_item(tcx, def_id).rebase_onto(
                        tcx,
                        impl_def_id.to_def_id(),
                        tcx.impl_trait_ref(impl_def_id).unwrap().instantiate_identity().args,
                    );
                    tcx.arena.alloc_from_iter(
                        ty::EarlyBinder::bind(tcx.assumed_wf_types_for_rpitit(rpitit_def_id))
                            .iter_instantiated_copied(tcx, args)
                            .chain(tcx.assumed_wf_types(impl_def_id).into_iter().copied()),
                    )
                }
            }
        }
        DefKind::AssocConst | DefKind::AssocTy => tcx.assumed_wf_types(tcx.local_parent(def_id)),
        DefKind::OpaqueTy => bug!("implied bounds are not defined for opaques"),
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::TyParam
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody => ty::List::empty(),
    }
}

fn fn_sig_spans(tcx: TyCtxt<'_>, def_id: LocalDefId) -> impl Iterator<Item = Span> {
    let node = tcx.hir_node_by_def_id(def_id);
    if let Some(decl) = node.fn_decl() {
        decl.inputs.iter().map(|ty| ty.span).chain(iter::once(decl.output.span()))
    } else {
        bug!("unexpected item for fn {def_id:?}: {node:?}")
    }
}

fn impl_spans(tcx: TyCtxt<'_>, def_id: LocalDefId) -> impl Iterator<Item = Span> {
    let item = tcx.hir_expect_item(def_id);
    if let hir::ItemKind::Impl(impl_) = item.kind {
        let trait_args = impl_
            .of_trait
            .into_iter()
            .flat_map(|trait_ref| trait_ref.path.segments.last().unwrap().args().args)
            .map(|arg| arg.span());
        let dummy_spans_for_default_args =
            impl_.of_trait.into_iter().flat_map(|trait_ref| iter::repeat(trait_ref.path.span));
        iter::once(impl_.self_ty.span).chain(trait_args).chain(dummy_spans_for_default_args)
    } else {
        bug!("unexpected item for impl {def_id:?}: {item:?}")
    }
}
