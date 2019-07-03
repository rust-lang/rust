//! Partitioning Codegen Units for Incremental Compilation
//! ======================================================
//!
//! The task of this module is to take the complete set of monomorphizations of
//! a crate and produce a set of codegen units from it, where a codegen unit
//! is a named set of (mono-item, linkage) pairs. That is, this module
//! decides which monomorphization appears in which codegen units with which
//! linkage. The following paragraphs describe some of the background on the
//! partitioning scheme.
//!
//! The most important opportunity for saving on compilation time with
//! incremental compilation is to avoid re-codegenning and re-optimizing code.
//! Since the unit of codegen and optimization for LLVM is "modules" or, how
//! we call them "codegen units", the particulars of how much time can be saved
//! by incremental compilation are tightly linked to how the output program is
//! partitioned into these codegen units prior to passing it to LLVM --
//! especially because we have to treat codegen units as opaque entities once
//! they are created: There is no way for us to incrementally update an existing
//! LLVM module and so we have to build any such module from scratch if it was
//! affected by some change in the source code.
//!
//! From that point of view it would make sense to maximize the number of
//! codegen units by, for example, putting each function into its own module.
//! That way only those modules would have to be re-compiled that were actually
//! affected by some change, minimizing the number of functions that could have
//! been re-used but just happened to be located in a module that is
//! re-compiled.
//!
//! However, since LLVM optimization does not work across module boundaries,
//! using such a highly granular partitioning would lead to very slow runtime
//! code since it would effectively prohibit inlining and other inter-procedure
//! optimizations. We want to avoid that as much as possible.
//!
//! Thus we end up with a trade-off: The bigger the codegen units, the better
//! LLVM's optimizer can do its work, but also the smaller the compilation time
//! reduction we get from incremental compilation.
//!
//! Ideally, we would create a partitioning such that there are few big codegen
//! units with few interdependencies between them. For now though, we use the
//! following heuristic to determine the partitioning:
//!
//! - There are two codegen units for every source-level module:
//! - One for "stable", that is non-generic, code
//! - One for more "volatile" code, i.e., monomorphized instances of functions
//!   defined in that module
//!
//! In order to see why this heuristic makes sense, let's take a look at when a
//! codegen unit can get invalidated:
//!
//! 1. The most straightforward case is when the BODY of a function or global
//! changes. Then any codegen unit containing the code for that item has to be
//! re-compiled. Note that this includes all codegen units where the function
//! has been inlined.
//!
//! 2. The next case is when the SIGNATURE of a function or global changes. In
//! this case, all codegen units containing a REFERENCE to that item have to be
//! re-compiled. This is a superset of case 1.
//!
//! 3. The final and most subtle case is when a REFERENCE to a generic function
//! is added or removed somewhere. Even though the definition of the function
//! might be unchanged, a new REFERENCE might introduce a new monomorphized
//! instance of this function which has to be placed and compiled somewhere.
//! Conversely, when removing a REFERENCE, it might have been the last one with
//! that particular set of generic arguments and thus we have to remove it.
//!
//! From the above we see that just using one codegen unit per source-level
//! module is not such a good idea, since just adding a REFERENCE to some
//! generic item somewhere else would invalidate everything within the module
//! containing the generic item. The heuristic above reduces this detrimental
//! side-effect of references a little by at least not touching the non-generic
//! code of the module.
//!
//! A Note on Inlining
//! ------------------
//! As briefly mentioned above, in order for LLVM to be able to inline a
//! function call, the body of the function has to be available in the LLVM
//! module where the call is made. This has a few consequences for partitioning:
//!
//! - The partitioning algorithm has to take care of placing functions into all
//!   codegen units where they should be available for inlining. It also has to
//!   decide on the correct linkage for these functions.
//!
//! - The partitioning algorithm has to know which functions are likely to get
//!   inlined, so it can distribute function instantiations accordingly. Since
//!   there is no way of knowing for sure which functions LLVM will decide to
//!   inline in the end, we apply a heuristic here: Only functions marked with
//!   `#[inline]` are considered for inlining by the partitioner. The current
//!   implementation will not try to determine if a function is likely to be
//!   inlined by looking at the functions definition.
//!
//! Note though that as a side-effect of creating a codegen units per
//! source-level module, functions from the same module will be available for
//! inlining, even when they are not marked `#[inline]`.

use std::collections::hash_map::Entry;
use std::cmp;
use std::sync::Arc;

use syntax::symbol::InternedString;
use rustc::hir::CodegenFnAttrFlags;
use rustc::hir::def::DefKind;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE, CRATE_DEF_INDEX};
use rustc::mir::mono::{Linkage, Visibility, CodegenUnitNameBuilder, CodegenUnit};
use rustc::middle::exported_symbols::SymbolExportLevel;
use rustc::ty::{self, DefIdTree, TyCtxt, InstanceDef};
use rustc::ty::print::characteristic_def_id_of_type;
use rustc::ty::query::Providers;
use rustc::util::common::time;
use rustc::util::nodemap::{DefIdSet, FxHashMap, FxHashSet};
use rustc::mir::mono::{MonoItem, InstantiationMode};

use crate::monomorphize::collector::InliningMap;
use crate::monomorphize::collector::{self, MonoItemCollectionMode};

pub enum PartitioningStrategy {
    /// Generates one codegen unit per source-level module.
    PerModule,

    /// Partition the whole crate into a fixed number of codegen units.
    FixedUnitCount(usize)
}

// Anything we can't find a proper codegen unit for goes into this.
fn fallback_cgu_name(name_builder: &mut CodegenUnitNameBuilder<'_>) -> InternedString {
    name_builder.build_cgu_name(LOCAL_CRATE, &["fallback"], Some("cgu"))
}

pub fn partition<'tcx, I>(
    tcx: TyCtxt<'tcx>,
    mono_items: I,
    strategy: PartitioningStrategy,
    inlining_map: &InliningMap<'tcx>,
) -> Vec<CodegenUnit<'tcx>>
where
    I: Iterator<Item = MonoItem<'tcx>>,
{
    // In the first step, we place all regular monomorphizations into their
    // respective 'home' codegen unit. Regular monomorphizations are all
    // functions and statics defined in the local crate.
    let mut initial_partitioning = place_root_mono_items(tcx, mono_items);

    initial_partitioning.codegen_units.iter_mut().for_each(|cgu| cgu.estimate_size(tcx));

    debug_dump(tcx, "INITIAL PARTITIONING:", initial_partitioning.codegen_units.iter());

    // If the partitioning should produce a fixed count of codegen units, merge
    // until that count is reached.
    if let PartitioningStrategy::FixedUnitCount(count) = strategy {
        merge_codegen_units(tcx, &mut initial_partitioning, count);

        debug_dump(tcx, "POST MERGING:", initial_partitioning.codegen_units.iter());
    }

    // In the next step, we use the inlining map to determine which additional
    // monomorphizations have to go into each codegen unit. These additional
    // monomorphizations can be drop-glue, functions from external crates, and
    // local functions the definition of which is marked with `#[inline]`.
    let mut post_inlining = place_inlined_mono_items(initial_partitioning,
                                                            inlining_map);

    post_inlining.codegen_units.iter_mut().for_each(|cgu| cgu.estimate_size(tcx));

    debug_dump(tcx, "POST INLINING:", post_inlining.codegen_units.iter());

    // Next we try to make as many symbols "internal" as possible, so LLVM has
    // more freedom to optimize.
    if !tcx.sess.opts.cg.link_dead_code {
        internalize_symbols(tcx, &mut post_inlining, inlining_map);
    }

    // Finally, sort by codegen unit name, so that we get deterministic results.
    let PostInliningPartitioning {
        codegen_units: mut result,
        mono_item_placements: _,
        internalization_candidates: _,
    } = post_inlining;

    result.sort_by(|cgu1, cgu2| {
        cgu1.name().cmp(cgu2.name())
    });

    result
}

struct PreInliningPartitioning<'tcx> {
    codegen_units: Vec<CodegenUnit<'tcx>>,
    roots: FxHashSet<MonoItem<'tcx>>,
    internalization_candidates: FxHashSet<MonoItem<'tcx>>,
}

/// For symbol internalization, we need to know whether a symbol/mono-item is
/// accessed from outside the codegen unit it is defined in. This type is used
/// to keep track of that.
#[derive(Clone, PartialEq, Eq, Debug)]
enum MonoItemPlacement {
    SingleCgu { cgu_name: InternedString },
    MultipleCgus,
}

struct PostInliningPartitioning<'tcx> {
    codegen_units: Vec<CodegenUnit<'tcx>>,
    mono_item_placements: FxHashMap<MonoItem<'tcx>, MonoItemPlacement>,
    internalization_candidates: FxHashSet<MonoItem<'tcx>>,
}

fn place_root_mono_items<'tcx, I>(tcx: TyCtxt<'tcx>, mono_items: I) -> PreInliningPartitioning<'tcx>
where
    I: Iterator<Item = MonoItem<'tcx>>,
{
    let mut roots = FxHashSet::default();
    let mut codegen_units = FxHashMap::default();
    let is_incremental_build = tcx.sess.opts.incremental.is_some();
    let mut internalization_candidates = FxHashSet::default();

    // Determine if monomorphizations instantiated in this crate will be made
    // available to downstream crates. This depends on whether we are in
    // share-generics mode and whether the current crate can even have
    // downstream crates.
    let export_generics = tcx.sess.opts.share_generics() &&
                          tcx.local_crate_exports_generics();

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
    let cgu_name_cache = &mut FxHashMap::default();

    for mono_item in mono_items {
        match mono_item.instantiation_mode(tcx) {
            InstantiationMode::GloballyShared { .. } => {}
            InstantiationMode::LocalCopy => continue,
        }

        let characteristic_def_id = characteristic_def_id_of_mono_item(tcx, mono_item);
        let is_volatile = is_incremental_build &&
                          mono_item.is_generic_fn();

        let codegen_unit_name = match characteristic_def_id {
            Some(def_id) => compute_codegen_unit_name(tcx,
                                                      cgu_name_builder,
                                                      def_id,
                                                      is_volatile,
                                                      cgu_name_cache),
            None => fallback_cgu_name(cgu_name_builder),
        };

        let codegen_unit = codegen_units.entry(codegen_unit_name.clone())
            .or_insert_with(|| CodegenUnit::new(codegen_unit_name.clone()));

        let mut can_be_internalized = true;
        let (linkage, visibility) = mono_item_linkage_and_visibility(
            tcx,
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
        codegen_units.insert(codegen_unit_name.clone(),
                             CodegenUnit::new(codegen_unit_name.clone()));
    }

    PreInliningPartitioning {
        codegen_units: codegen_units.into_iter()
                                    .map(|(_, codegen_unit)| codegen_unit)
                                    .collect(),
        roots,
        internalization_candidates,
    }
}

fn mono_item_linkage_and_visibility(
    tcx: TyCtxt<'tcx>,
    mono_item: &MonoItem<'tcx>,
    can_be_internalized: &mut bool,
    export_generics: bool,
) -> (Linkage, Visibility) {
    if let Some(explicit_linkage) = mono_item.explicit_linkage(tcx) {
        return (explicit_linkage, Visibility::Default)
    }
    let vis = mono_item_visibility(
        tcx,
        mono_item,
        can_be_internalized,
        export_generics,
    );
    (Linkage::External, vis)
}

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
        MonoItem::GlobalAsm(hir_id) => {
            let def_id = tcx.hir().local_def_id_from_hir_id(*hir_id);
            return if tcx.is_reachable_non_generic(def_id) {
                *can_be_internalized = false;
                default_visibility(tcx, def_id, false)
            } else {
                Visibility::Hidden
            };
        }
    };

    let def_id = match instance.def {
        InstanceDef::Item(def_id) => def_id,

        // These are all compiler glue and such, never exported, always hidden.
        InstanceDef::VtableShim(..) |
        InstanceDef::FnPtrShim(..) |
        InstanceDef::Virtual(..) |
        InstanceDef::Intrinsic(..) |
        InstanceDef::ClosureOnceShim { .. } |
        InstanceDef::DropGlue(..) |
        InstanceDef::CloneShim(..) => {
            return Visibility::Hidden
        }
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
        return Visibility::Hidden
    }

    let is_generic = instance.substs.non_erasable_generics().next().is_some();

    // Upstream `DefId` instances get different handling than local ones.
    if !def_id.is_local() {
        return if export_generics && is_generic {
            // If it is a upstream monomorphization and we export generics, we must make
            // it available to downstream crates.
            *can_be_internalized = false;
            default_visibility(tcx, def_id, true)
        } else {
            Visibility::Hidden
        }
    }

    if is_generic {
        if export_generics {
            if tcx.is_unreachable_local_definition(def_id) {
                // This instance cannot be used from another crate.
                Visibility::Hidden
            } else {
                // This instance might be useful in a downstream crate.
                *can_be_internalized = false;
                default_visibility(tcx, def_id, true)
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
        if tcx.is_reachable_non_generic(def_id) {
            *can_be_internalized = false;
            debug_assert!(!is_generic);
            return default_visibility(tcx, def_id, false)
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
    if !tcx.sess.target.target.options.default_hidden_visibility {
        return Visibility::Default
    }

    // Generic functions never have export-level C.
    if is_generic {
        return Visibility::Hidden
    }

    // Things with export level C don't get instantiated in
    // downstream crates.
    if !id.is_local() {
        return Visibility::Hidden
    }

    // C-export level items remain at `Default`, all other internal
    // items become `Hidden`.
    match tcx.reachable_non_generics(id.krate).get(&id) {
        Some(SymbolExportLevel::C) => Visibility::Default,
        _ => Visibility::Hidden,
    }
}

fn merge_codegen_units<'tcx>(
    tcx: TyCtxt<'tcx>,
    initial_partitioning: &mut PreInliningPartitioning<'tcx>,
    target_cgu_count: usize,
) {
    assert!(target_cgu_count >= 1);
    let codegen_units = &mut initial_partitioning.codegen_units;

    // Note that at this point in time the `codegen_units` here may not be in a
    // deterministic order (but we know they're deterministically the same set).
    // We want this merging to produce a deterministic ordering of codegen units
    // from the input.
    //
    // Due to basically how we've implemented the merging below (merge the two
    // smallest into each other) we're sure to start off with a deterministic
    // order (sorted by name). This'll mean that if two cgus have the same size
    // the stable sort below will keep everything nice and deterministic.
    codegen_units.sort_by_key(|cgu| *cgu.name());

    // Merge the two smallest codegen units until the target size is reached.
    while codegen_units.len() > target_cgu_count {
        // Sort small cgus to the back
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        let mut smallest = codegen_units.pop().unwrap();
        let second_smallest = codegen_units.last_mut().unwrap();

        second_smallest.modify_size_estimate(smallest.size_estimate());
        for (k, v) in smallest.items_mut().drain() {
            second_smallest.items_mut().insert(k, v);
        }
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(tcx);
    for (index, cgu) in codegen_units.iter_mut().enumerate() {
        cgu.set_name(numbered_codegen_unit_name(cgu_name_builder, index));
    }
}

fn place_inlined_mono_items<'tcx>(initial_partitioning: PreInliningPartitioning<'tcx>,
                                  inlining_map: &InliningMap<'tcx>)
                                  -> PostInliningPartitioning<'tcx> {
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
            follow_inlining(*root, inlining_map, &mut reachable);
        }

        let mut new_codegen_unit = CodegenUnit::new(old_codegen_unit.name().clone());

        // Add all monomorphizations that are not already there.
        for mono_item in reachable {
            if let Some(linkage) = old_codegen_unit.items().get(&mono_item) {
                // This is a root, just copy it over.
                new_codegen_unit.items_mut().insert(mono_item, *linkage);
            } else {
                if roots.contains(&mono_item) {
                    bug!("GloballyShared mono-item inlined into other CGU: \
                          {:?}", mono_item);
                }

                // This is a CGU-private copy.
                new_codegen_unit.items_mut().insert(
                    mono_item,
                    (Linkage::Internal, Visibility::Default),
                );
            }

            if !single_codegen_unit {
                // If there is more than one codegen unit, we need to keep track
                // in which codegen units each monomorphization is placed.
                match mono_item_placements.entry(mono_item) {
                    Entry::Occupied(e) => {
                        let placement = e.into_mut();
                        debug_assert!(match *placement {
                            MonoItemPlacement::SingleCgu { ref cgu_name } => {
                                *cgu_name != *new_codegen_unit.name()
                            }
                            MonoItemPlacement::MultipleCgus => true,
                        });
                        *placement = MonoItemPlacement::MultipleCgus;
                    }
                    Entry::Vacant(e) => {
                        e.insert(MonoItemPlacement::SingleCgu {
                            cgu_name: new_codegen_unit.name().clone()
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

    fn follow_inlining<'tcx>(mono_item: MonoItem<'tcx>,
                             inlining_map: &InliningMap<'tcx>,
                             visited: &mut FxHashSet<MonoItem<'tcx>>) {
        if !visited.insert(mono_item) {
            return;
        }

        inlining_map.with_inlining_candidates(mono_item, |target| {
            follow_inlining(target, inlining_map, visited);
        });
    }
}

fn internalize_symbols<'tcx>(
    _tcx: TyCtxt<'tcx>,
    partitioning: &mut PostInliningPartitioning<'tcx>,
    inlining_map: &InliningMap<'tcx>,
) {
    if partitioning.codegen_units.len() == 1 {
        // Fast path for when there is only one codegen unit. In this case we
        // can internalize all candidates, since there is nowhere else they
        // could be accessed from.
        for cgu in &mut partitioning.codegen_units {
            for candidate in &partitioning.internalization_candidates {
                cgu.items_mut().insert(*candidate,
                                       (Linkage::Internal, Visibility::Default));
            }
        }

        return;
    }

    // Build a map from every monomorphization to all the monomorphizations that
    // reference it.
    let mut accessor_map: FxHashMap<MonoItem<'tcx>, Vec<MonoItem<'tcx>>> = Default::default();
    inlining_map.iter_accesses(|accessor, accessees| {
        for accessee in accessees {
            accessor_map.entry(*accessee)
                        .or_default()
                        .push(accessor);
        }
    });

    let mono_item_placements = &partitioning.mono_item_placements;

    // For each internalization candidates in each codegen unit, check if it is
    // accessed from outside its defining codegen unit.
    for cgu in &mut partitioning.codegen_units {
        let home_cgu = MonoItemPlacement::SingleCgu {
            cgu_name: cgu.name().clone()
        };

        for (accessee, linkage_and_visibility) in cgu.items_mut() {
            if !partitioning.internalization_candidates.contains(accessee) {
                // This item is no candidate for internalizing, so skip it.
                continue
            }
            debug_assert_eq!(mono_item_placements[accessee], home_cgu);

            if let Some(accessors) = accessor_map.get(accessee) {
                if accessors.iter()
                            .filter_map(|accessor| {
                                // Some accessors might not have been
                                // instantiated. We can safely ignore those.
                                mono_item_placements.get(accessor)
                            })
                            .any(|placement| *placement != home_cgu) {
                    // Found an accessor from another CGU, so skip to the next
                    // item without marking this one as internal.
                    continue
                }
            }

            // If we got here, we did not find any accesses from other CGUs,
            // so it's fine to make this monomorphization internal.
            *linkage_and_visibility = (Linkage::Internal, Visibility::Default);
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
                ty::InstanceDef::Item(def_id) => def_id,
                ty::InstanceDef::VtableShim(..) |
                ty::InstanceDef::FnPtrShim(..) |
                ty::InstanceDef::ClosureOnceShim { .. } |
                ty::InstanceDef::Intrinsic(..) |
                ty::InstanceDef::DropGlue(..) |
                ty::InstanceDef::Virtual(..) |
                ty::InstanceDef::CloneShim(..) => return None
            };

            // If this is a method, we want to put it into the same module as
            // its self-type. If the self-type does not provide a characteristic
            // DefId, we use the location of the impl after all.

            if tcx.trait_of_item(def_id).is_some() {
                let self_ty = instance.substs.type_at(0);
                // This is an implementation of a trait method.
                return characteristic_def_id_of_type(self_ty).or(Some(def_id));
            }

            if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
                // This is a method within an inherent impl, find out what the
                // self-type is:
                let impl_self_ty = tcx.subst_and_normalize_erasing_regions(
                    instance.substs,
                    ty::ParamEnv::reveal_all(),
                    &tcx.type_of(impl_def_id),
                );
                if let Some(def_id) = characteristic_def_id_of_type(impl_self_ty) {
                    return Some(def_id);
                }
            }

            Some(def_id)
        }
        MonoItem::Static(def_id) => Some(def_id),
        MonoItem::GlobalAsm(hir_id) => Some(tcx.hir().local_def_id_from_hir_id(hir_id)),
    }
}

type CguNameCache = FxHashMap<(DefId, bool), InternedString>;

fn compute_codegen_unit_name(
    tcx: TyCtxt<'_>,
    name_builder: &mut CodegenUnitNameBuilder<'_>,
    def_id: DefId,
    volatile: bool,
    cache: &mut CguNameCache,
) -> InternedString {
    // Find the innermost module that is not nested within a function.
    let mut current_def_id = def_id;
    let mut cgu_def_id = None;
    // Walk backwards from the item we want to find the module for.
    loop {
        if current_def_id.index == CRATE_DEF_INDEX {
            if cgu_def_id.is_none() {
                // If we have not found a module yet, take the crate root.
                cgu_def_id = Some(DefId {
                    krate: def_id.krate,
                    index: CRATE_DEF_INDEX,
                });
            }
            break
        } else if tcx.def_kind(current_def_id) == Some(DefKind::Mod) {
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

    cache.entry((cgu_def_id, volatile)).or_insert_with(|| {
        let def_path = tcx.def_path(cgu_def_id);

        let components = def_path
            .data
            .iter()
            .map(|part| part.data.as_interned_str());

        let volatile_suffix = if volatile {
            Some("volatile")
        } else {
            None
        };

        name_builder.build_cgu_name(def_path.krate, components, volatile_suffix)
    }).clone()
}

fn numbered_codegen_unit_name(
    name_builder: &mut CodegenUnitNameBuilder<'_>,
    index: usize,
) -> InternedString {
    name_builder.build_cgu_name_no_mangle(LOCAL_CRATE, &["cgu"], Some(index))
}

fn debug_dump<'a, 'tcx, I>(tcx: TyCtxt<'tcx>, label: &str, cgus: I)
where
    I: Iterator<Item = &'a CodegenUnit<'tcx>>,
    'tcx: 'a,
{
    if cfg!(debug_assertions) {
        debug!("{}", label);
        for cgu in cgus {
            debug!("CodegenUnit {}:", cgu.name());

            for (mono_item, linkage) in cgu.items() {
                let symbol_name = mono_item.symbol_name(tcx).as_str();
                let symbol_hash_start = symbol_name.rfind('h');
                let symbol_hash = symbol_hash_start.map(|i| &symbol_name[i ..])
                                                   .unwrap_or("<no hash>");

                debug!(" - {} [{:?}] [{}]",
                       mono_item.to_string(tcx, true),
                       linkage,
                       symbol_hash);
            }

            debug!("");
        }
    }
}

#[inline(never)] // give this a place in the profiler
fn assert_symbols_are_distinct<'a, 'tcx, I>(tcx: TyCtxt<'tcx>, mono_items: I)
where
    I: Iterator<Item = &'a MonoItem<'tcx>>,
    'tcx: 'a,
{
    let mut symbols: Vec<_> = mono_items.map(|mono_item| {
        (mono_item, mono_item.symbol_name(tcx))
    }).collect();

    symbols.sort_by_key(|sym| sym.1);

    for pair in symbols.windows(2) {
        let sym1 = &pair[0].1;
        let sym2 = &pair[1].1;

        if sym1 == sym2 {
            let mono_item1 = pair[0].0;
            let mono_item2 = pair[1].0;

            let span1 = mono_item1.local_span(tcx);
            let span2 = mono_item2.local_span(tcx);

            // Deterministically select one of the spans for error reporting
            let span = match (span1, span2) {
                (Some(span1), Some(span2)) => {
                    Some(if span1.lo().0 > span2.lo().0 {
                        span1
                    } else {
                        span2
                    })
                }
                (span1, span2) => span1.or(span2),
            };

            let error_message = format!("symbol `{}` is already defined", sym1);

            if let Some(span) = span {
                tcx.sess.span_fatal(span, &error_message)
            } else {
                tcx.sess.fatal(&error_message)
            }
        }
    }
}

fn collect_and_partition_mono_items(
    tcx: TyCtxt<'_>,
    cnum: CrateNum,
) -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'_>>>>) {
    assert_eq!(cnum, LOCAL_CRATE);

    let collection_mode = match tcx.sess.opts.debugging_opts.print_mono_items {
        Some(ref s) => {
            let mode_string = s.to_lowercase();
            let mode_string = mode_string.trim();
            if mode_string == "eager" {
                MonoItemCollectionMode::Eager
            } else {
                if mode_string != "lazy" {
                    let message = format!("Unknown codegen-item collection mode '{}'. \
                                           Falling back to 'lazy' mode.",
                                          mode_string);
                    tcx.sess.warn(&message);
                }

                MonoItemCollectionMode::Lazy
            }
        }
        None => {
            if tcx.sess.opts.cg.link_dead_code {
                MonoItemCollectionMode::Eager
            } else {
                MonoItemCollectionMode::Lazy
            }
        }
    };

    let (items, inlining_map) =
        time(tcx.sess, "monomorphization collection", || {
            collector::collect_crate_mono_items(tcx, collection_mode)
    });

    tcx.sess.abort_if_errors();

    assert_symbols_are_distinct(tcx, items.iter());

    let strategy = if tcx.sess.opts.incremental.is_some() {
        PartitioningStrategy::PerModule
    } else {
        PartitioningStrategy::FixedUnitCount(tcx.sess.codegen_units())
    };

    let codegen_units = time(tcx.sess, "codegen unit partitioning", || {
        partition(
            tcx,
            items.iter().cloned(),
            strategy,
            &inlining_map
        )
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>()
    });

    let mono_items: DefIdSet = items.iter().filter_map(|mono_item| {
        match *mono_item {
            MonoItem::Fn(ref instance) => Some(instance.def_id()),
            MonoItem::Static(def_id) => Some(def_id),
            _ => None,
        }
    }).collect();

    if tcx.sess.opts.debugging_opts.print_mono_items.is_some() {
        let mut item_to_cgus: FxHashMap<_, Vec<_>> = Default::default();

        for cgu in &codegen_units {
            for (&mono_item, &linkage) in cgu.items() {
                item_to_cgus.entry(mono_item)
                            .or_default()
                            .push((cgu.name().clone(), linkage));
            }
        }

        let mut item_keys: Vec<_> = items
            .iter()
            .map(|i| {
                let mut output = i.to_string(tcx, false);
                output.push_str(" @@");
                let mut empty = Vec::new();
                let cgus = item_to_cgus.get_mut(i).unwrap_or(&mut empty);
                cgus.sort_by_key(|(name, _)| *name);
                cgus.dedup();
                for &(ref cgu_name, (linkage, _)) in cgus.iter() {
                    output.push_str(" ");
                    output.push_str(&cgu_name.as_str());

                    let linkage_abbrev = match linkage {
                        Linkage::External => "External",
                        Linkage::AvailableExternally => "Available",
                        Linkage::LinkOnceAny => "OnceAny",
                        Linkage::LinkOnceODR => "OnceODR",
                        Linkage::WeakAny => "WeakAny",
                        Linkage::WeakODR => "WeakODR",
                        Linkage::Appending => "Appending",
                        Linkage::Internal => "Internal",
                        Linkage::Private => "Private",
                        Linkage::ExternalWeak => "ExternalWeak",
                        Linkage::Common => "Common",
                    };

                    output.push_str("[");
                    output.push_str(linkage_abbrev);
                    output.push_str("]");
                }
                output
            })
            .collect();

        item_keys.sort();

        for item in item_keys {
            println!("MONO_ITEM {}", item);
        }
    }

    (Arc::new(mono_items), Arc::new(codegen_units))
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.collect_and_partition_mono_items =
        collect_and_partition_mono_items;

    providers.is_codegened_item = |tcx, def_id| {
        let (all_mono_items, _) =
            tcx.collect_and_partition_mono_items(LOCAL_CRATE);
        all_mono_items.contains(&def_id)
    };

    providers.codegen_unit = |tcx, name| {
        let (_, all) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);
        all.iter()
            .find(|cgu| *cgu.name() == name)
            .cloned()
            .unwrap_or_else(|| panic!("failed to find cgu with name {:?}", name))
    };
}
