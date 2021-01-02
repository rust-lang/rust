use std::collections::hash_map::Entry;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::DefPathDataName;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::SymbolExportLevel;
use rustc_middle::mir::mono::{CodegenUnit, CodegenUnitNameBuilder, Linkage, Visibility};
use rustc_middle::mir::mono::{InstantiationMode, MonoItem};
use rustc_middle::ty::print::characteristic_def_id_of_type;
use rustc_middle::ty::{self, DefIdTree, InstanceDef, TyCtxt};
use rustc_span::symbol::Symbol;

use super::PartitioningCx;
use crate::collector::InliningMap;
use crate::partitioning::merging;
use crate::partitioning::{
    MonoItemPlacement, Partitioner, PostInliningPartitioning, PreInliningPartitioning,
};

pub struct DefaultPartitioning;

impl<'tcx> Partitioner<'tcx> for DefaultPartitioning {
    fn place_root_mono_items(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        mono_items: &mut dyn Iterator<Item = MonoItem<'tcx>>,
    ) -> PreInliningPartitioning<'tcx> {
        let mut roots = FxHashSet::default();
        let mut codegen_units = FxHashMap::default();
        let is_incremental_build = cx.tcx.sess.opts.incremental.is_some();
        let mut internalization_candidates = FxHashSet::default();

        // Determine if monomorphizations instantiated in this crate will be made
        // available to downstream crates. This depends on whether we are in
        // share-generics mode and whether the current crate can even have
        // downstream crates.
        let export_generics =
            cx.tcx.sess.opts.share_generics() && cx.tcx.local_crate_exports_generics();

        let cgu_name_builder = &mut CodegenUnitNameBuilder::new(cx.tcx);
        let cgu_name_cache = &mut FxHashMap::default();

        for mono_item in mono_items {
            match mono_item.instantiation_mode(cx.tcx) {
                InstantiationMode::GloballyShared { .. } => {}
                InstantiationMode::LocalCopy => continue,
            }

            let characteristic_def_id = characteristic_def_id_of_mono_item(cx.tcx, mono_item);
            let is_volatile = is_incremental_build && mono_item.is_generic_fn();

            let codegen_unit_name = match characteristic_def_id {
                Some(def_id) => compute_codegen_unit_name(
                    cx.tcx,
                    cgu_name_builder,
                    def_id,
                    is_volatile,
                    cgu_name_cache,
                ),
                None => fallback_cgu_name(cgu_name_builder),
            };

            let codegen_unit = codegen_units
                .entry(codegen_unit_name)
                .or_insert_with(|| CodegenUnit::new(codegen_unit_name));

            let mut can_be_internalized = true;
            let (linkage, visibility) = mono_item_linkage_and_visibility(
                cx.tcx,
                &mono_item,
                &mut can_be_internalized,
                export_generics,
            );
            if visibility == Visibility::Hidden && can_be_internalized {
                internalization_candidates.insert(mono_item);
            }

            codegen_unit.items_mut().insert(mono_item, (linkage, visibility));
            roots.insert(mono_item);
        }

        // Always ensure we have at least one CGU; otherwise, if we have a
        // crate with just types (for example), we could wind up with no CGU.
        if codegen_units.is_empty() {
            let codegen_unit_name = fallback_cgu_name(cgu_name_builder);
            codegen_units.insert(codegen_unit_name, CodegenUnit::new(codegen_unit_name));
        }

        PreInliningPartitioning {
            codegen_units: codegen_units
                .into_iter()
                .map(|(_, codegen_unit)| codegen_unit)
                .collect(),
            roots,
            internalization_candidates,
        }
    }

    fn merge_codegen_units(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        initial_partitioning: &mut PreInliningPartitioning<'tcx>,
    ) {
        merging::merge_codegen_units(cx, initial_partitioning);
    }

    fn place_inlined_mono_items(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        initial_partitioning: PreInliningPartitioning<'tcx>,
    ) -> PostInliningPartitioning<'tcx> {
        let mut new_partitioning = Vec::new();
        let mut mono_item_placements = FxHashMap::default();

        let PreInliningPartitioning {
            codegen_units: initial_cgus,
            roots,
            internalization_candidates,
        } = initial_partitioning;

        let single_codegen_unit = initial_cgus.len() == 1;

        for old_codegen_unit in initial_cgus {
            // Collect all items that need to be available in this codegen unit.
            let mut reachable = FxHashSet::default();
            for root in old_codegen_unit.items().keys() {
                follow_inlining(*root, cx.inlining_map, &mut reachable);
            }

            let mut new_codegen_unit = CodegenUnit::new(old_codegen_unit.name());

            // Add all monomorphizations that are not already there.
            for mono_item in reachable {
                if let Some(linkage) = old_codegen_unit.items().get(&mono_item) {
                    // This is a root, just copy it over.
                    new_codegen_unit.items_mut().insert(mono_item, *linkage);
                } else {
                    if roots.contains(&mono_item) {
                        bug!(
                            "GloballyShared mono-item inlined into other CGU: \
                              {:?}",
                            mono_item
                        );
                    }

                    // This is a CGU-private copy.
                    new_codegen_unit
                        .items_mut()
                        .insert(mono_item, (Linkage::Internal, Visibility::Default));
                }

                if !single_codegen_unit {
                    // If there is more than one codegen unit, we need to keep track
                    // in which codegen units each monomorphization is placed.
                    match mono_item_placements.entry(mono_item) {
                        Entry::Occupied(e) => {
                            let placement = e.into_mut();
                            debug_assert!(match *placement {
                                MonoItemPlacement::SingleCgu { cgu_name } => {
                                    cgu_name != new_codegen_unit.name()
                                }
                                MonoItemPlacement::MultipleCgus => true,
                            });
                            *placement = MonoItemPlacement::MultipleCgus;
                        }
                        Entry::Vacant(e) => {
                            e.insert(MonoItemPlacement::SingleCgu {
                                cgu_name: new_codegen_unit.name(),
                            });
                        }
                    }
                }
            }

            new_partitioning.push(new_codegen_unit);
        }

        return PostInliningPartitioning {
            codegen_units: new_partitioning,
            mono_item_placements,
            internalization_candidates,
        };

        fn follow_inlining<'tcx>(
            mono_item: MonoItem<'tcx>,
            inlining_map: &InliningMap<'tcx>,
            visited: &mut FxHashSet<MonoItem<'tcx>>,
        ) {
            if !visited.insert(mono_item) {
                return;
            }

            inlining_map.with_inlining_candidates(mono_item, |target| {
                follow_inlining(target, inlining_map, visited);
            });
        }
    }

    fn internalize_symbols(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        partitioning: &mut PostInliningPartitioning<'tcx>,
    ) {
        if partitioning.codegen_units.len() == 1 {
            // Fast path for when there is only one codegen unit. In this case we
            // can internalize all candidates, since there is nowhere else they
            // could be accessed from.
            for cgu in &mut partitioning.codegen_units {
                for candidate in &partitioning.internalization_candidates {
                    cgu.items_mut().insert(*candidate, (Linkage::Internal, Visibility::Default));
                }
            }

            return;
        }

        // Build a map from every monomorphization to all the monomorphizations that
        // reference it.
        let mut accessor_map: FxHashMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>> = Default::default();
        cx.inlining_map.iter_accesses(|accessor, accessees| {
            for accessee in accessees {
                accessor_map.entry(*accessee).or_default().push(accessor);
            }
        });

        let mono_item_placements = &partitioning.mono_item_placements;

        // For each internalization candidates in each codegen unit, check if it is
        // accessed from outside its defining codegen unit.
        for cgu in &mut partitioning.codegen_units {
            let home_cgu = MonoItemPlacement::SingleCgu { cgu_name: cgu.name() };

            for (accessee, linkage_and_visibility) in cgu.items_mut() {
                if !partitioning.internalization_candidates.contains(accessee) {
                    // This item is no candidate for internalizing, so skip it.
                    continue;
                }
                debug_assert_eq!(mono_item_placements[accessee], home_cgu);

                if let Some(accessors) = accessor_map.get(accessee) {
                    if accessors
                        .iter()
                        .filter_map(|accessor| {
                            // Some accessors might not have been
                            // instantiated. We can safely ignore those.
                            mono_item_placements.get(accessor)
                        })
                        .any(|placement| *placement != home_cgu)
                    {
                        // Found an accessor from another CGU, so skip to the next
                        // item without marking this one as internal.
                        continue;
                    }
                }

                // If we got here, we did not find any accesses from other CGUs,
                // so it's fine to make this monomorphization internal.
                *linkage_and_visibility = (Linkage::Internal, Visibility::Default);
            }
        }
    }
}

fn characteristic_def_id_of_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_item: MonoItem<'tcx>,
) -> Option<DefId> {
    match mono_item {
        MonoItem::Fn(instance) => {
            let def_id = match instance.def {
                ty::InstanceDef::Item(def) => def.did,
                ty::InstanceDef::VtableShim(..)
                | ty::InstanceDef::ReifyShim(..)
                | ty::InstanceDef::FnPtrShim(..)
                | ty::InstanceDef::ClosureOnceShim { .. }
                | ty::InstanceDef::Intrinsic(..)
                | ty::InstanceDef::DropGlue(..)
                | ty::InstanceDef::Virtual(..)
                | ty::InstanceDef::CloneShim(..) => return None,
            };

            // If this is a method, we want to put it into the same module as
            // its self-type. If the self-type does not provide a characteristic
            // DefId, we use the location of the impl after all.

            if tcx.trait_of_item(def_id).is_some() {
                let self_ty = instance.substs.type_at(0);
                // This is a default implementation of a trait method.
                return characteristic_def_id_of_type(self_ty).or(Some(def_id));
            }

            if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
                if tcx.sess.opts.incremental.is_some()
                    && tcx.trait_id_of_impl(impl_def_id) == tcx.lang_items().drop_trait()
                {
                    // Put `Drop::drop` into the same cgu as `drop_in_place`
                    // since `drop_in_place` is the only thing that can
                    // call it.
                    return None;
                }
                // This is a method within an impl, find out what the self-type is:
                let impl_self_ty = tcx.subst_and_normalize_erasing_regions(
                    instance.substs,
                    ty::ParamEnv::reveal_all(),
                    tcx.type_of(impl_def_id),
                );
                if let Some(def_id) = characteristic_def_id_of_type(impl_self_ty) {
                    return Some(def_id);
                }
            }

            Some(def_id)
        }
        MonoItem::Static(def_id) => Some(def_id),
        MonoItem::GlobalAsm(item_id) => Some(item_id.def_id.to_def_id()),
    }
}

fn compute_codegen_unit_name(
    tcx: TyCtxt<'_>,
    name_builder: &mut CodegenUnitNameBuilder<'_>,
    def_id: DefId,
    volatile: bool,
    cache: &mut CguNameCache,
) -> Symbol {
    // Find the innermost module that is not nested within a function.
    let mut current_def_id = def_id;
    let mut cgu_def_id = None;
    // Walk backwards from the item we want to find the module for.
    loop {
        if current_def_id.index == CRATE_DEF_INDEX {
            if cgu_def_id.is_none() {
                // If we have not found a module yet, take the crate root.
                cgu_def_id = Some(DefId { krate: def_id.krate, index: CRATE_DEF_INDEX });
            }
            break;
        } else if tcx.def_kind(current_def_id) == DefKind::Mod {
            if cgu_def_id.is_none() {
                cgu_def_id = Some(current_def_id);
            }
        } else {
            // If we encounter something that is not a module, throw away
            // any module that we've found so far because we now know that
            // it is nested within something else.
            cgu_def_id = None;
        }

        current_def_id = tcx.parent(current_def_id).unwrap();
    }

    let cgu_def_id = cgu_def_id.unwrap();

    *cache.entry((cgu_def_id, volatile)).or_insert_with(|| {
        let def_path = tcx.def_path(cgu_def_id);

        let components = def_path.data.iter().map(|part| match part.data.name() {
            DefPathDataName::Named(name) => name,
            DefPathDataName::Anon { .. } => unreachable!(),
        });

        let volatile_suffix = volatile.then_some("volatile");

        name_builder.build_cgu_name(def_path.krate, components, volatile_suffix)
    })
}

// Anything we can't find a proper codegen unit for goes into this.
fn fallback_cgu_name(name_builder: &mut CodegenUnitNameBuilder<'_>) -> Symbol {
    name_builder.build_cgu_name(LOCAL_CRATE, &["fallback"], Some("cgu"))
}

fn mono_item_linkage_and_visibility(
    tcx: TyCtxt<'tcx>,
    mono_item: &MonoItem<'tcx>,
    can_be_internalized: &mut bool,
    export_generics: bool,
) -> (Linkage, Visibility) {
    if let Some(explicit_linkage) = mono_item.explicit_linkage(tcx) {
        return (explicit_linkage, Visibility::Default);
    }
    let vis = mono_item_visibility(tcx, mono_item, can_be_internalized, export_generics);
    (Linkage::External, vis)
}

type CguNameCache = FxHashMap<(DefId, bool), Symbol>;

fn mono_item_visibility(
    tcx: TyCtxt<'tcx>,
    mono_item: &MonoItem<'tcx>,
    can_be_internalized: &mut bool,
    export_generics: bool,
) -> Visibility {
    let instance = match mono_item {
        // This is pretty complicated; see below.
        MonoItem::Fn(instance) => instance,

        // Misc handling for generics and such, but otherwise:
        MonoItem::Static(def_id) => {
            return if tcx.is_reachable_non_generic(*def_id) {
                *can_be_internalized = false;
                default_visibility(tcx, *def_id, false)
            } else {
                Visibility::Hidden
            };
        }
        MonoItem::GlobalAsm(item_id) => {
            return if tcx.is_reachable_non_generic(item_id.def_id) {
                *can_be_internalized = false;
                default_visibility(tcx, item_id.def_id.to_def_id(), false)
            } else {
                Visibility::Hidden
            };
        }
    };

    let def_id = match instance.def {
        InstanceDef::Item(def) => def.did,
        InstanceDef::DropGlue(def_id, Some(_)) => def_id,

        // These are all compiler glue and such, never exported, always hidden.
        InstanceDef::VtableShim(..)
        | InstanceDef::ReifyShim(..)
        | InstanceDef::FnPtrShim(..)
        | InstanceDef::Virtual(..)
        | InstanceDef::Intrinsic(..)
        | InstanceDef::ClosureOnceShim { .. }
        | InstanceDef::DropGlue(..)
        | InstanceDef::CloneShim(..) => return Visibility::Hidden,
    };

    // The `start_fn` lang item is actually a monomorphized instance of a
    // function in the standard library, used for the `main` function. We don't
    // want to export it so we tag it with `Hidden` visibility but this symbol
    // is only referenced from the actual `main` symbol which we unfortunately
    // don't know anything about during partitioning/collection. As a result we
    // forcibly keep this symbol out of the `internalization_candidates` set.
    //
    // FIXME: eventually we don't want to always force this symbol to have
    //        hidden visibility, it should indeed be a candidate for
    //        internalization, but we have to understand that it's referenced
    //        from the `main` symbol we'll generate later.
    //
    //        This may be fixable with a new `InstanceDef` perhaps? Unsure!
    if tcx.lang_items().start_fn() == Some(def_id) {
        *can_be_internalized = false;
        return Visibility::Hidden;
    }

    let is_generic = instance.substs.non_erasable_generics().next().is_some();

    // Upstream `DefId` instances get different handling than local ones.
    let def_id = if let Some(def_id) = def_id.as_local() {
        def_id
    } else {
        return if export_generics && is_generic {
            // If it is an upstream monomorphization and we export generics, we must make
            // it available to downstream crates.
            *can_be_internalized = false;
            default_visibility(tcx, def_id, true)
        } else {
            Visibility::Hidden
        };
    };

    if is_generic {
        if export_generics {
            if tcx.is_unreachable_local_definition(def_id) {
                // This instance cannot be used from another crate.
                Visibility::Hidden
            } else {
                // This instance might be useful in a downstream crate.
                *can_be_internalized = false;
                default_visibility(tcx, def_id.to_def_id(), true)
            }
        } else {
            // We are not exporting generics or the definition is not reachable
            // for downstream crates, we can internalize its instantiations.
            Visibility::Hidden
        }
    } else {
        // If this isn't a generic function then we mark this a `Default` if
        // this is a reachable item, meaning that it's a symbol other crates may
        // access when they link to us.
        if tcx.is_reachable_non_generic(def_id.to_def_id()) {
            *can_be_internalized = false;
            debug_assert!(!is_generic);
            return default_visibility(tcx, def_id.to_def_id(), false);
        }

        // If this isn't reachable then we're gonna tag this with `Hidden`
        // visibility. In some situations though we'll want to prevent this
        // symbol from being internalized.
        //
        // There's two categories of items here:
        //
        // * First is weak lang items. These are basically mechanisms for
        //   libcore to forward-reference symbols defined later in crates like
        //   the standard library or `#[panic_handler]` definitions. The
        //   definition of these weak lang items needs to be referenceable by
        //   libcore, so we're no longer a candidate for internalization.
        //   Removal of these functions can't be done by LLVM but rather must be
        //   done by the linker as it's a non-local decision.
        //
        // * Second is "std internal symbols". Currently this is primarily used
        //   for allocator symbols. Allocators are a little weird in their
        //   implementation, but the idea is that the compiler, at the last
        //   minute, defines an allocator with an injected object file. The
        //   `alloc` crate references these symbols (`__rust_alloc`) and the
        //   definition doesn't get hooked up until a linked crate artifact is
        //   generated.
        //
        //   The symbols synthesized by the compiler (`__rust_alloc`) are thin
        //   veneers around the actual implementation, some other symbol which
        //   implements the same ABI. These symbols (things like `__rg_alloc`,
        //   `__rdl_alloc`, `__rde_alloc`, etc), are all tagged with "std
        //   internal symbols".
        //
        //   The std-internal symbols here **should not show up in a dll as an
        //   exported interface**, so they return `false` from
        //   `is_reachable_non_generic` above and we'll give them `Hidden`
        //   visibility below. Like the weak lang items, though, we can't let
        //   LLVM internalize them as this decision is left up to the linker to
        //   omit them, so prevent them from being internalized.
        let attrs = tcx.codegen_fn_attrs(def_id);
        if attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
            *can_be_internalized = false;
        }

        Visibility::Hidden
    }
}

fn default_visibility(tcx: TyCtxt<'_>, id: DefId, is_generic: bool) -> Visibility {
    if !tcx.sess.target.default_hidden_visibility {
        return Visibility::Default;
    }

    // Generic functions never have export-level C.
    if is_generic {
        return Visibility::Hidden;
    }

    // Things with export level C don't get instantiated in
    // downstream crates.
    if !id.is_local() {
        return Visibility::Hidden;
    }

    // C-export level items remain at `Default`, all other internal
    // items become `Hidden`.
    match tcx.reachable_non_generics(id.krate).get(&id) {
        Some(SymbolExportLevel::C) => Visibility::Default,
        _ => Visibility::Hidden,
    }
}
