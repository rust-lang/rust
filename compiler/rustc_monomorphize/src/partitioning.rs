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

use std::cmp;
use std::collections::hash_map::Entry;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdSet, LOCAL_CRATE};
use rustc_hir::definitions::DefPathDataName;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_middle::mir;
use rustc_middle::mir::mono::{
    CodegenUnit, CodegenUnitNameBuilder, InstantiationMode, Linkage, MonoItem, Visibility,
};
use rustc_middle::query::Providers;
use rustc_middle::ty::print::{characteristic_def_id_of_type, with_no_trimmed_paths};
use rustc_middle::ty::{self, visit::TypeVisitableExt, InstanceDef, TyCtxt};
use rustc_session::config::{DumpMonoStatsFormat, SwitchWithOptPath};
use rustc_session::CodegenUnits;
use rustc_span::symbol::Symbol;

use crate::collector::UsageMap;
use crate::collector::{self, MonoItemCollectionMode};
use crate::errors::{CouldntDumpMonoStats, SymbolAlreadyDefined, UnknownCguCollectionMode};

struct PartitioningCx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    usage_map: &'a UsageMap<'tcx>,
}

struct PlacedMonoItems<'tcx> {
    /// The codegen units, sorted by name to make things deterministic.
    codegen_units: Vec<CodegenUnit<'tcx>>,

    internalization_candidates: FxHashSet<MonoItem<'tcx>>,

    /// These must be obtained when the iterator in `partition` runs. They
    /// can't be obtained later because some inlined functions might not be
    /// reachable.
    unique_inlined_stats: (usize, usize),
}

// The output CGUs are sorted by name.
fn partition<'tcx, I>(
    tcx: TyCtxt<'tcx>,
    mono_items: I,
    usage_map: &UsageMap<'tcx>,
) -> Vec<CodegenUnit<'tcx>>
where
    I: Iterator<Item = MonoItem<'tcx>>,
{
    let _prof_timer = tcx.prof.generic_activity("cgu_partitioning");

    let cx = &PartitioningCx { tcx, usage_map };

    // Place all mono items into a codegen unit. `place_mono_items` is
    // responsible for initializing the CGU size estimates.
    let PlacedMonoItems { mut codegen_units, internalization_candidates, unique_inlined_stats } = {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_place_items");
        let placed = place_mono_items(cx, mono_items);

        debug_dump(tcx, "PLACE", &placed.codegen_units, placed.unique_inlined_stats);

        placed
    };

    // Merge until we have at most `max_cgu_count` codegen units.
    // `merge_codegen_units` is responsible for updating the CGU size
    // estimates.
    {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_merge_cgus");
        merge_codegen_units(cx, &mut codegen_units);
        debug_dump(tcx, "MERGE", &codegen_units, unique_inlined_stats);
    }

    // Make as many symbols "internal" as possible, so LLVM has more freedom to
    // optimize.
    if !tcx.sess.link_dead_code() {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_internalize_symbols");
        internalize_symbols(cx, &mut codegen_units, internalization_candidates);

        debug_dump(tcx, "INTERNALIZE", &codegen_units, unique_inlined_stats);
    }

    // Mark one CGU for dead code, if necessary.
    let instrument_dead_code =
        tcx.sess.instrument_coverage() && !tcx.sess.instrument_coverage_except_unused_functions();
    if instrument_dead_code {
        mark_code_coverage_dead_code_cgu(&mut codegen_units);
    }

    // Ensure CGUs are sorted by name, so that we get deterministic results.
    assert!(codegen_units.is_sorted_by(|a, b| Some(a.name().as_str().cmp(b.name().as_str()))));

    codegen_units
}

fn place_mono_items<'tcx, I>(cx: &PartitioningCx<'_, 'tcx>, mono_items: I) -> PlacedMonoItems<'tcx>
where
    I: Iterator<Item = MonoItem<'tcx>>,
{
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

    let mut num_unique_inlined_items = 0;
    let mut unique_inlined_items_size = 0;
    for mono_item in mono_items {
        // Handle only root items directly here. Inlined items are handled at
        // the bottom of the loop based on reachability.
        match mono_item.instantiation_mode(cx.tcx) {
            InstantiationMode::GloballyShared { .. } => {}
            InstantiationMode::LocalCopy => {
                num_unique_inlined_items += 1;
                unique_inlined_items_size += mono_item.size_estimate(cx.tcx);
                continue;
            }
        }

        let characteristic_def_id = characteristic_def_id_of_mono_item(cx.tcx, mono_item);
        let is_volatile = is_incremental_build && mono_item.is_generic_fn();

        let cgu_name = match characteristic_def_id {
            Some(def_id) => compute_codegen_unit_name(
                cx.tcx,
                cgu_name_builder,
                def_id,
                is_volatile,
                cgu_name_cache,
            ),
            None => fallback_cgu_name(cgu_name_builder),
        };

        let cgu = codegen_units.entry(cgu_name).or_insert_with(|| CodegenUnit::new(cgu_name));

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

        cgu.items_mut().insert(mono_item, (linkage, visibility));

        // Get all inlined items that are reachable from `mono_item` without
        // going via another root item. This includes drop-glue, functions from
        // external crates, and local functions the definition of which is
        // marked with `#[inline]`.
        let mut reachable_inlined_items = FxHashSet::default();
        get_reachable_inlined_items(cx.tcx, mono_item, cx.usage_map, &mut reachable_inlined_items);

        // Add those inlined items. It's possible an inlined item is reachable
        // from multiple root items within a CGU, which is fine, it just means
        // the `insert` will be a no-op.
        for inlined_item in reachable_inlined_items {
            // This is a CGU-private copy.
            cgu.items_mut().insert(inlined_item, (Linkage::Internal, Visibility::Default));
        }
    }

    // Always ensure we have at least one CGU; otherwise, if we have a
    // crate with just types (for example), we could wind up with no CGU.
    if codegen_units.is_empty() {
        let cgu_name = fallback_cgu_name(cgu_name_builder);
        codegen_units.insert(cgu_name, CodegenUnit::new(cgu_name));
    }

    let mut codegen_units: Vec<_> = codegen_units.into_values().collect();
    codegen_units.sort_by(|a, b| a.name().as_str().cmp(b.name().as_str()));

    for cgu in codegen_units.iter_mut() {
        cgu.compute_size_estimate(cx.tcx);
    }

    return PlacedMonoItems {
        codegen_units,
        internalization_candidates,
        unique_inlined_stats: (num_unique_inlined_items, unique_inlined_items_size),
    };

    fn get_reachable_inlined_items<'tcx>(
        tcx: TyCtxt<'tcx>,
        item: MonoItem<'tcx>,
        usage_map: &UsageMap<'tcx>,
        visited: &mut FxHashSet<MonoItem<'tcx>>,
    ) {
        usage_map.for_each_inlined_used_item(tcx, item, |inlined_item| {
            let is_new = visited.insert(inlined_item);
            if is_new {
                get_reachable_inlined_items(tcx, inlined_item, usage_map, visited);
            }
        });
    }
}

// This function requires the CGUs to be sorted by name on input, and ensures
// they are sorted by name on return, for deterministic behaviour.
fn merge_codegen_units<'tcx>(
    cx: &PartitioningCx<'_, 'tcx>,
    codegen_units: &mut Vec<CodegenUnit<'tcx>>,
) {
    assert!(cx.tcx.sess.codegen_units().as_usize() >= 1);

    // A sorted order here ensures merging is deterministic.
    assert!(codegen_units.is_sorted_by(|a, b| Some(a.name().as_str().cmp(b.name().as_str()))));

    // This map keeps track of what got merged into what.
    let mut cgu_contents: FxHashMap<Symbol, Vec<Symbol>> =
        codegen_units.iter().map(|cgu| (cgu.name(), vec![cgu.name()])).collect();

    // Having multiple CGUs can drastically speed up compilation. But for
    // non-incremental builds, tiny CGUs slow down compilation *and* result in
    // worse generated code. So we don't allow CGUs smaller than this (unless
    // there is just one CGU, of course). Note that CGU sizes of 100,000+ are
    // common in larger programs, so this isn't all that large.
    const NON_INCR_MIN_CGU_SIZE: usize = 1800;

    // Repeatedly merge the two smallest codegen units as long as:
    // - we have more CGUs than the upper limit, or
    // - (Non-incremental builds only) the user didn't specify a CGU count, and
    //   there are multiple CGUs, and some are below the minimum size.
    //
    // The "didn't specify a CGU count" condition is because when an explicit
    // count is requested we observe it as closely as possible. For example,
    // the `compiler_builtins` crate sets `codegen-units = 10000` and it's
    // critical they aren't merged. Also, some tests use explicit small values
    // and likewise won't work if small CGUs are merged.
    while codegen_units.len() > cx.tcx.sess.codegen_units().as_usize()
        || (cx.tcx.sess.opts.incremental.is_none()
            && matches!(cx.tcx.sess.codegen_units(), CodegenUnits::Default(_))
            && codegen_units.len() > 1
            && codegen_units.iter().any(|cgu| cgu.size_estimate() < NON_INCR_MIN_CGU_SIZE))
    {
        // Sort small cgus to the back.
        codegen_units.sort_by_cached_key(|cgu| cmp::Reverse(cgu.size_estimate()));

        let mut smallest = codegen_units.pop().unwrap();
        let second_smallest = codegen_units.last_mut().unwrap();

        // Move the items from `smallest` to `second_smallest`. Some of them
        // may be duplicate inlined items, in which case the destination CGU is
        // unaffected. Recalculate size estimates afterwards.
        second_smallest.items_mut().extend(smallest.items_mut().drain());
        second_smallest.compute_size_estimate(cx.tcx);

        // Record that `second_smallest` now contains all the stuff that was
        // in `smallest` before.
        let mut consumed_cgu_names = cgu_contents.remove(&smallest.name()).unwrap();
        cgu_contents.get_mut(&second_smallest.name()).unwrap().append(&mut consumed_cgu_names);

        debug!(
            "CodegenUnit {} merged into CodegenUnit {}",
            smallest.name(),
            second_smallest.name()
        );
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(cx.tcx);

    // Rename the newly merged CGUs.
    if cx.tcx.sess.opts.incremental.is_some() {
        // If we are doing incremental compilation, we want CGU names to
        // reflect the path of the source level module they correspond to.
        // For CGUs that contain the code of multiple modules because of the
        // merging done above, we use a concatenation of the names of all
        // contained CGUs.
        let new_cgu_names: FxHashMap<Symbol, String> = cgu_contents
            .into_iter()
            // This `filter` makes sure we only update the name of CGUs that
            // were actually modified by merging.
            .filter(|(_, cgu_contents)| cgu_contents.len() > 1)
            .map(|(current_cgu_name, cgu_contents)| {
                let mut cgu_contents: Vec<&str> = cgu_contents.iter().map(|s| s.as_str()).collect();

                // Sort the names, so things are deterministic and easy to
                // predict. We are sorting primitive `&str`s here so we can
                // use unstable sort.
                cgu_contents.sort_unstable();

                (current_cgu_name, cgu_contents.join("--"))
            })
            .collect();

        for cgu in codegen_units.iter_mut() {
            if let Some(new_cgu_name) = new_cgu_names.get(&cgu.name()) {
                if cx.tcx.sess.opts.unstable_opts.human_readable_cgu_names {
                    cgu.set_name(Symbol::intern(&new_cgu_name));
                } else {
                    // If we don't require CGU names to be human-readable,
                    // we use a fixed length hash of the composite CGU name
                    // instead.
                    let new_cgu_name = CodegenUnit::mangle_name(&new_cgu_name);
                    cgu.set_name(Symbol::intern(&new_cgu_name));
                }
            }
        }

        // A sorted order here ensures what follows can be deterministic.
        codegen_units.sort_by(|a, b| a.name().as_str().cmp(b.name().as_str()));
    } else {
        // When compiling non-incrementally, we rename the CGUS so they have
        // identical names except for the numeric suffix, something like
        // `regex.f10ba03eb5ec7975-cgu.N`, where `N` varies.
        //
        // It is useful for debugging and profiling purposes if the resulting
        // CGUs are sorted by name *and* reverse sorted by size. (CGU 0 is the
        // biggest, CGU 1 is the second biggest, etc.)
        //
        // So first we reverse sort by size. Then we generate the names with
        // zero-padded suffixes, which means they are automatically sorted by
        // names. The numeric suffix width depends on the number of CGUs, which
        // is always greater than zero:
        // - [1,9]     CGUS: `0`, `1`, `2`, ...
        // - [10,99]   CGUS: `00`, `01`, `02`, ...
        // - [100,999] CGUS: `000`, `001`, `002`, ...
        // - etc.
        //
        // If we didn't zero-pad the sorted-by-name order would be `XYZ-cgu.0`,
        // `XYZ-cgu.1`, `XYZ-cgu.10`, `XYZ-cgu.11`, ..., `XYZ-cgu.2`, etc.
        codegen_units.sort_by_key(|cgu| cmp::Reverse(cgu.size_estimate()));
        let num_digits = codegen_units.len().ilog10() as usize + 1;
        for (index, cgu) in codegen_units.iter_mut().enumerate() {
            // Note: `WorkItem::short_description` depends on this name ending
            // with `-cgu.` followed by a numeric suffix. Please keep it in
            // sync with this code.
            let suffix = format!("{index:0num_digits$}");
            let numbered_codegen_unit_name =
                cgu_name_builder.build_cgu_name_no_mangle(LOCAL_CRATE, &["cgu"], Some(suffix));
            cgu.set_name(numbered_codegen_unit_name);
        }
    }
}

fn internalize_symbols<'tcx>(
    cx: &PartitioningCx<'_, 'tcx>,
    codegen_units: &mut [CodegenUnit<'tcx>],
    internalization_candidates: FxHashSet<MonoItem<'tcx>>,
) {
    /// For symbol internalization, we need to know whether a symbol/mono-item
    /// is used from outside the codegen unit it is defined in. This type is
    /// used to keep track of that.
    #[derive(Clone, PartialEq, Eq, Debug)]
    enum MonoItemPlacement {
        SingleCgu { cgu_name: Symbol },
        MultipleCgus,
    }

    let mut mono_item_placements = FxHashMap::default();
    let single_codegen_unit = codegen_units.len() == 1;

    if !single_codegen_unit {
        for cgu in codegen_units.iter_mut() {
            for item in cgu.items().keys() {
                // If there is more than one codegen unit, we need to keep track
                // in which codegen units each monomorphization is placed.
                match mono_item_placements.entry(*item) {
                    Entry::Occupied(e) => {
                        let placement = e.into_mut();
                        debug_assert!(match *placement {
                            MonoItemPlacement::SingleCgu { cgu_name } => cgu_name != cgu.name(),
                            MonoItemPlacement::MultipleCgus => true,
                        });
                        *placement = MonoItemPlacement::MultipleCgus;
                    }
                    Entry::Vacant(e) => {
                        e.insert(MonoItemPlacement::SingleCgu { cgu_name: cgu.name() });
                    }
                }
            }
        }
    }

    // For each internalization candidates in each codegen unit, check if it is
    // used from outside its defining codegen unit.
    for cgu in codegen_units {
        let home_cgu = MonoItemPlacement::SingleCgu { cgu_name: cgu.name() };

        for (item, linkage_and_visibility) in cgu.items_mut() {
            if !internalization_candidates.contains(item) {
                // This item is no candidate for internalizing, so skip it.
                continue;
            }

            if !single_codegen_unit {
                debug_assert_eq!(mono_item_placements[item], home_cgu);

                if let Some(user_items) = cx.usage_map.get_user_items(*item) {
                    if user_items
                        .iter()
                        .filter_map(|user_item| {
                            // Some user mono items might not have been
                            // instantiated. We can safely ignore those.
                            mono_item_placements.get(user_item)
                        })
                        .any(|placement| *placement != home_cgu)
                    {
                        // Found a user from another CGU, so skip to the next item
                        // without marking this one as internal.
                        continue;
                    }
                }
            }

            // If we got here, we did not find any uses from other CGUs, so
            // it's fine to make this monomorphization internal.
            *linkage_and_visibility = (Linkage::Internal, Visibility::Default);
        }
    }
}

fn mark_code_coverage_dead_code_cgu<'tcx>(codegen_units: &mut [CodegenUnit<'tcx>]) {
    assert!(!codegen_units.is_empty());

    // Find the smallest CGU that has exported symbols and put the dead
    // function stubs in that CGU. We look for exported symbols to increase
    // the likelihood the linker won't throw away the dead functions.
    // FIXME(#92165): In order to truly resolve this, we need to make sure
    // the object file (CGU) containing the dead function stubs is included
    // in the final binary. This will probably require forcing these
    // function symbols to be included via `-u` or `/include` linker args.
    let dead_code_cgu = codegen_units
        .iter_mut()
        .filter(|cgu| cgu.items().iter().any(|(_, (linkage, _))| *linkage == Linkage::External))
        .min_by_key(|cgu| cgu.size_estimate());

    // If there are no CGUs that have externally linked items, then we just
    // pick the first CGU as a fallback.
    let dead_code_cgu = if let Some(cgu) = dead_code_cgu { cgu } else { &mut codegen_units[0] };

    dead_code_cgu.make_code_coverage_dead_code_cgu();
}

fn characteristic_def_id_of_mono_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_item: MonoItem<'tcx>,
) -> Option<DefId> {
    match mono_item {
        MonoItem::Fn(instance) => {
            let def_id = match instance.def {
                ty::InstanceDef::Item(def) => def,
                ty::InstanceDef::VTableShim(..)
                | ty::InstanceDef::ReifyShim(..)
                | ty::InstanceDef::FnPtrShim(..)
                | ty::InstanceDef::ClosureOnceShim { .. }
                | ty::InstanceDef::Intrinsic(..)
                | ty::InstanceDef::DropGlue(..)
                | ty::InstanceDef::Virtual(..)
                | ty::InstanceDef::CloneShim(..)
                | ty::InstanceDef::ThreadLocalShim(..)
                | ty::InstanceDef::FnPtrAddrShim(..) => return None,
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

                // When polymorphization is enabled, methods which do not depend on their generic
                // parameters, but the self-type of their impl block do will fail to normalize.
                if !tcx.sess.opts.unstable_opts.polymorphize || !instance.has_param() {
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
            }

            Some(def_id)
        }
        MonoItem::Static(def_id) => Some(def_id),
        MonoItem::GlobalAsm(item_id) => Some(item_id.owner_id.to_def_id()),
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
        if current_def_id.is_crate_root() {
            if cgu_def_id.is_none() {
                // If we have not found a module yet, take the crate root.
                cgu_def_id = Some(def_id.krate.as_def_id());
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

        current_def_id = tcx.parent(current_def_id);
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

fn mono_item_linkage_and_visibility<'tcx>(
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

fn static_visibility<'tcx>(
    tcx: TyCtxt<'tcx>,
    can_be_internalized: &mut bool,
    def_id: DefId,
) -> Visibility {
    if tcx.is_reachable_non_generic(def_id) {
        *can_be_internalized = false;
        default_visibility(tcx, def_id, false)
    } else {
        Visibility::Hidden
    }
}

fn mono_item_visibility<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_item: &MonoItem<'tcx>,
    can_be_internalized: &mut bool,
    export_generics: bool,
) -> Visibility {
    let instance = match mono_item {
        // This is pretty complicated; see below.
        MonoItem::Fn(instance) => instance,

        // Misc handling for generics and such, but otherwise:
        MonoItem::Static(def_id) => return static_visibility(tcx, can_be_internalized, *def_id),
        MonoItem::GlobalAsm(item_id) => {
            return static_visibility(tcx, can_be_internalized, item_id.owner_id.to_def_id());
        }
    };

    let def_id = match instance.def {
        InstanceDef::Item(def_id) | InstanceDef::DropGlue(def_id, Some(_)) => def_id,

        // We match the visibility of statics here
        InstanceDef::ThreadLocalShim(def_id) => {
            return static_visibility(tcx, can_be_internalized, def_id);
        }

        // These are all compiler glue and such, never exported, always hidden.
        InstanceDef::VTableShim(..)
        | InstanceDef::ReifyShim(..)
        | InstanceDef::FnPtrShim(..)
        | InstanceDef::Virtual(..)
        | InstanceDef::Intrinsic(..)
        | InstanceDef::ClosureOnceShim { .. }
        | InstanceDef::DropGlue(..)
        | InstanceDef::CloneShim(..)
        | InstanceDef::FnPtrAddrShim(..) => return Visibility::Hidden,
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
    let Some(def_id) = def_id.as_local() else {
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
        // use when they link to us.
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
        //   definition of these weak lang items needs to be referencable by
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
        Some(SymbolExportInfo { level: SymbolExportLevel::C, .. }) => Visibility::Default,
        _ => Visibility::Hidden,
    }
}

fn debug_dump<'a, 'tcx: 'a>(
    tcx: TyCtxt<'tcx>,
    label: &str,
    cgus: &[CodegenUnit<'tcx>],
    (unique_inlined_items, unique_inlined_size): (usize, usize),
) {
    let dump = move || {
        use std::fmt::Write;

        let mut num_cgus = 0;
        let mut all_cgu_sizes = Vec::new();

        // Note: every unique root item is placed exactly once, so the number
        // of unique root items always equals the number of placed root items.

        let mut root_items = 0;
        // unique_inlined_items is passed in above.
        let mut placed_inlined_items = 0;

        let mut root_size = 0;
        // unique_inlined_size is passed in above.
        let mut placed_inlined_size = 0;

        for cgu in cgus.iter() {
            num_cgus += 1;
            all_cgu_sizes.push(cgu.size_estimate());

            for (item, _) in cgu.items() {
                match item.instantiation_mode(tcx) {
                    InstantiationMode::GloballyShared { .. } => {
                        root_items += 1;
                        root_size += item.size_estimate(tcx);
                    }
                    InstantiationMode::LocalCopy => {
                        placed_inlined_items += 1;
                        placed_inlined_size += item.size_estimate(tcx);
                    }
                }
            }
        }

        all_cgu_sizes.sort_unstable_by_key(|&n| cmp::Reverse(n));

        let unique_items = root_items + unique_inlined_items;
        let placed_items = root_items + placed_inlined_items;
        let items_ratio = placed_items as f64 / unique_items as f64;

        let unique_size = root_size + unique_inlined_size;
        let placed_size = root_size + placed_inlined_size;
        let size_ratio = placed_size as f64 / unique_size as f64;

        let mean_cgu_size = placed_size as f64 / num_cgus as f64;

        assert_eq!(placed_size, all_cgu_sizes.iter().sum::<usize>());

        let s = &mut String::new();
        let _ = writeln!(s, "{label}");
        let _ = writeln!(
            s,
            "- unique items: {unique_items} ({root_items} root + {unique_inlined_items} inlined), \
               unique size: {unique_size} ({root_size} root + {unique_inlined_size} inlined)\n\
             - placed items: {placed_items} ({root_items} root + {placed_inlined_items} inlined), \
               placed size: {placed_size} ({root_size} root + {placed_inlined_size} inlined)\n\
             - placed/unique items ratio: {items_ratio:.2}, \
               placed/unique size ratio: {size_ratio:.2}\n\
             - CGUs: {num_cgus}, mean size: {mean_cgu_size:.1}, sizes: {}",
            list(&all_cgu_sizes),
        );
        let _ = writeln!(s);

        for (i, cgu) in cgus.iter().enumerate() {
            let name = cgu.name();
            let size = cgu.size_estimate();
            let num_items = cgu.items().len();
            let mean_size = size as f64 / num_items as f64;

            let mut placed_item_sizes: Vec<_> =
                cgu.items().iter().map(|(item, _)| item.size_estimate(tcx)).collect();
            placed_item_sizes.sort_unstable_by_key(|&n| cmp::Reverse(n));
            let sizes = list(&placed_item_sizes);

            let _ = writeln!(s, "- CGU[{i}]");
            let _ = writeln!(s, "  - {name}, size: {size}");
            let _ =
                writeln!(s, "  - items: {num_items}, mean size: {mean_size:.1}, sizes: {sizes}",);

            for (item, linkage) in cgu.items_in_deterministic_order(tcx) {
                let symbol_name = item.symbol_name(tcx).name;
                let symbol_hash_start = symbol_name.rfind('h');
                let symbol_hash = symbol_hash_start.map_or("<no hash>", |i| &symbol_name[i..]);
                let size = item.size_estimate(tcx);
                let kind = match item.instantiation_mode(tcx) {
                    InstantiationMode::GloballyShared { .. } => "root",
                    InstantiationMode::LocalCopy => "inlined",
                };
                let _ = with_no_trimmed_paths!(writeln!(
                    s,
                    "  - {item} [{linkage:?}] [{symbol_hash}] ({kind}, size: {size})"
                ));
            }

            let _ = writeln!(s);
        }

        return std::mem::take(s);

        // Converts a slice to a string, capturing repetitions to save space.
        // E.g. `[4, 4, 4, 3, 2, 1, 1, 1, 1, 1]` -> "[4 (x3), 3, 2, 1 (x5)]".
        fn list(ns: &[usize]) -> String {
            let mut v = Vec::new();
            if ns.is_empty() {
                return "[]".to_string();
            }

            let mut elem = |curr, curr_count| {
                if curr_count == 1 {
                    v.push(format!("{curr}"));
                } else {
                    v.push(format!("{curr} (x{curr_count})"));
                }
            };

            let mut curr = ns[0];
            let mut curr_count = 1;

            for &n in &ns[1..] {
                if n != curr {
                    elem(curr, curr_count);
                    curr = n;
                    curr_count = 1;
                } else {
                    curr_count += 1;
                }
            }
            elem(curr, curr_count);

            let mut s = "[".to_string();
            s.push_str(&v.join(", "));
            s.push_str("]");
            s
        }
    };

    debug!("{}", dump());
}

#[inline(never)] // give this a place in the profiler
fn assert_symbols_are_distinct<'a, 'tcx, I>(tcx: TyCtxt<'tcx>, mono_items: I)
where
    I: Iterator<Item = &'a MonoItem<'tcx>>,
    'tcx: 'a,
{
    let _prof_timer = tcx.prof.generic_activity("assert_symbols_are_distinct");

    let mut symbols: Vec<_> =
        mono_items.map(|mono_item| (mono_item, mono_item.symbol_name(tcx))).collect();

    symbols.sort_by_key(|sym| sym.1);

    for &[(mono_item1, ref sym1), (mono_item2, ref sym2)] in symbols.array_windows() {
        if sym1 == sym2 {
            let span1 = mono_item1.local_span(tcx);
            let span2 = mono_item2.local_span(tcx);

            // Deterministically select one of the spans for error reporting
            let span = match (span1, span2) {
                (Some(span1), Some(span2)) => {
                    Some(if span1.lo().0 > span2.lo().0 { span1 } else { span2 })
                }
                (span1, span2) => span1.or(span2),
            };

            tcx.sess.emit_fatal(SymbolAlreadyDefined { span, symbol: sym1.to_string() });
        }
    }
}

fn collect_and_partition_mono_items(tcx: TyCtxt<'_>, (): ()) -> (&DefIdSet, &[CodegenUnit<'_>]) {
    let collection_mode = match tcx.sess.opts.unstable_opts.print_mono_items {
        Some(ref s) => {
            let mode = s.to_lowercase();
            let mode = mode.trim();
            if mode == "eager" {
                MonoItemCollectionMode::Eager
            } else {
                if mode != "lazy" {
                    tcx.sess.emit_warning(UnknownCguCollectionMode { mode });
                }

                MonoItemCollectionMode::Lazy
            }
        }
        None => {
            if tcx.sess.link_dead_code() {
                MonoItemCollectionMode::Eager
            } else {
                MonoItemCollectionMode::Lazy
            }
        }
    };

    let (items, usage_map) = collector::collect_crate_mono_items(tcx, collection_mode);

    tcx.sess.abort_if_errors();

    let (codegen_units, _) = tcx.sess.time("partition_and_assert_distinct_symbols", || {
        sync::join(
            || {
                let mut codegen_units = partition(tcx, items.iter().copied(), &usage_map);
                codegen_units[0].make_primary();
                &*tcx.arena.alloc_from_iter(codegen_units)
            },
            || assert_symbols_are_distinct(tcx, items.iter()),
        )
    });

    if tcx.prof.enabled() {
        // Record CGU size estimates for self-profiling.
        for cgu in codegen_units {
            tcx.prof.artifact_size(
                "codegen_unit_size_estimate",
                cgu.name().as_str(),
                cgu.size_estimate() as u64,
            );
        }
    }

    let mono_items: DefIdSet = items
        .iter()
        .filter_map(|mono_item| match *mono_item {
            MonoItem::Fn(ref instance) => Some(instance.def_id()),
            MonoItem::Static(def_id) => Some(def_id),
            _ => None,
        })
        .collect();

    // Output monomorphization stats per def_id
    if let SwitchWithOptPath::Enabled(ref path) = tcx.sess.opts.unstable_opts.dump_mono_stats {
        if let Err(err) =
            dump_mono_items_stats(tcx, &codegen_units, path, tcx.crate_name(LOCAL_CRATE))
        {
            tcx.sess.emit_fatal(CouldntDumpMonoStats { error: err.to_string() });
        }
    }

    if tcx.sess.opts.unstable_opts.print_mono_items.is_some() {
        let mut item_to_cgus: FxHashMap<_, Vec<_>> = Default::default();

        for cgu in codegen_units {
            for (&mono_item, &linkage) in cgu.items() {
                item_to_cgus.entry(mono_item).or_default().push((cgu.name(), linkage));
            }
        }

        let mut item_keys: Vec<_> = items
            .iter()
            .map(|i| {
                let mut output = with_no_trimmed_paths!(i.to_string());
                output.push_str(" @@");
                let mut empty = Vec::new();
                let cgus = item_to_cgus.get_mut(i).unwrap_or(&mut empty);
                cgus.sort_by_key(|(name, _)| *name);
                cgus.dedup();
                for &(ref cgu_name, (linkage, _)) in cgus.iter() {
                    output.push(' ');
                    output.push_str(cgu_name.as_str());

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

                    output.push('[');
                    output.push_str(linkage_abbrev);
                    output.push(']');
                }
                output
            })
            .collect();

        item_keys.sort();

        for item in item_keys {
            println!("MONO_ITEM {item}");
        }
    }

    (tcx.arena.alloc(mono_items), codegen_units)
}

/// Outputs stats about instantiation counts and estimated size, per `MonoItem`'s
/// def, to a file in the given output directory.
fn dump_mono_items_stats<'tcx>(
    tcx: TyCtxt<'tcx>,
    codegen_units: &[CodegenUnit<'tcx>],
    output_directory: &Option<PathBuf>,
    crate_name: Symbol,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_directory = if let Some(ref directory) = output_directory {
        fs::create_dir_all(directory)?;
        directory
    } else {
        Path::new(".")
    };

    let format = tcx.sess.opts.unstable_opts.dump_mono_stats_format;
    let ext = format.extension();
    let filename = format!("{crate_name}.mono_items.{ext}");
    let output_path = output_directory.join(&filename);
    let file = File::create(&output_path)?;
    let mut file = BufWriter::new(file);

    // Gather instantiated mono items grouped by def_id
    let mut items_per_def_id: FxHashMap<_, Vec<_>> = Default::default();
    for cgu in codegen_units {
        for (&mono_item, _) in cgu.items() {
            // Avoid variable-sized compiler-generated shims
            if mono_item.is_user_defined() {
                items_per_def_id.entry(mono_item.def_id()).or_default().push(mono_item);
            }
        }
    }

    #[derive(serde::Serialize)]
    struct MonoItem {
        name: String,
        instantiation_count: usize,
        size_estimate: usize,
        total_estimate: usize,
    }

    // Output stats sorted by total instantiated size, from heaviest to lightest
    let mut stats: Vec<_> = items_per_def_id
        .into_iter()
        .map(|(def_id, items)| {
            let name = with_no_trimmed_paths!(tcx.def_path_str(def_id));
            let instantiation_count = items.len();
            let size_estimate = items[0].size_estimate(tcx);
            let total_estimate = instantiation_count * size_estimate;
            MonoItem { name, instantiation_count, size_estimate, total_estimate }
        })
        .collect();
    stats.sort_unstable_by_key(|item| cmp::Reverse(item.total_estimate));

    if !stats.is_empty() {
        match format {
            DumpMonoStatsFormat::Json => serde_json::to_writer(file, &stats)?,
            DumpMonoStatsFormat::Markdown => {
                writeln!(
                    file,
                    "| Item | Instantiation count | Estimated Cost Per Instantiation | Total Estimated Cost |"
                )?;
                writeln!(file, "| --- | ---: | ---: | ---: |")?;

                for MonoItem { name, instantiation_count, size_estimate, total_estimate } in stats {
                    writeln!(
                        file,
                        "| `{name}` | {instantiation_count} | {size_estimate} | {total_estimate} |"
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn codegened_and_inlined_items(tcx: TyCtxt<'_>, (): ()) -> &DefIdSet {
    let (items, cgus) = tcx.collect_and_partition_mono_items(());
    let mut visited = DefIdSet::default();
    let mut result = items.clone();

    for cgu in cgus {
        for (item, _) in cgu.items() {
            if let MonoItem::Fn(ref instance) = item {
                let did = instance.def_id();
                if !visited.insert(did) {
                    continue;
                }
                let body = tcx.instance_mir(instance.def);
                for block in body.basic_blocks.iter() {
                    for statement in &block.statements {
                        let mir::StatementKind::Coverage(_) = statement.kind else { continue };
                        let scope = statement.source_info.scope;
                        if let Some(inlined) = scope.inlined_instance(&body.source_scopes) {
                            result.insert(inlined.def_id());
                        }
                    }
                }
            }
        }
    }

    tcx.arena.alloc(result)
}

pub fn provide(providers: &mut Providers) {
    providers.collect_and_partition_mono_items = collect_and_partition_mono_items;
    providers.codegened_and_inlined_items = codegened_and_inlined_items;

    providers.is_codegened_item = |tcx, def_id| {
        let (all_mono_items, _) = tcx.collect_and_partition_mono_items(());
        all_mono_items.contains(&def_id)
    };

    providers.codegen_unit = |tcx, name| {
        let (_, all) = tcx.collect_and_partition_mono_items(());
        all.iter()
            .find(|cgu| cgu.name() == name)
            .unwrap_or_else(|| panic!("failed to find cgu with name {name:?}"))
    };
}
