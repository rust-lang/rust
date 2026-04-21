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

use std::borrow::Cow;
use std::cmp;
use std::collections::hash_map::Entry;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::sync::par_join;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir::LangItem;
use rustc_hir::attrs::{InlineAttr, Linkage};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdSet, LOCAL_CRATE};
use rustc_hir::definitions::DefPathDataName;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_middle::mir::StatementKind;
use rustc_middle::mono::{
    CodegenUnit, CodegenUnitNameBuilder, InstantiationMode, LocalMonoItemCollection, MonoItem,
    MonoItemData, MonoItemPartitions, UsageMap, Visibility,
};
use rustc_middle::ty::print::{characteristic_def_id_of_type, with_no_trimmed_paths};
use rustc_middle::ty::trait_cast::{IntrinsicResolutions, TraitCastRequests};
use rustc_middle::ty::{self, InstanceKind, Ty, TyCtxt};
use rustc_middle::util::Providers;
use rustc_session::CodegenUnits;
use rustc_session::config::{DumpMonoStatsFormat, SwitchWithOptPath};
use rustc_span::Symbol;
use rustc_target::spec::SymbolVisibility;
use tracing::debug;

use crate::collector::{self, MonoItemCollectionStrategy};
use crate::erasure_safe::trait_metadata_index_outlives_class;
use crate::errors::{CouldntDumpMonoStats, SymbolAlreadyDefined};
use crate::graph_checks::target_specific_checks;

struct PartitioningCx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    usage_map: &'a UsageMap<'tcx>,
}

struct PlacedMonoItems<'tcx> {
    /// The codegen units, sorted by name to make things deterministic.
    codegen_units: Vec<CodegenUnit<'tcx>>,

    internalization_candidates: UnordSet<MonoItem<'tcx>>,
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
    let PlacedMonoItems { mut codegen_units, internalization_candidates } = {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_place_items");
        let placed = place_mono_items(cx, mono_items);

        debug_dump(tcx, "PLACE", &placed.codegen_units);

        placed
    };

    // Merge until we don't exceed the max CGU count.
    // `merge_codegen_units` is responsible for updating the CGU size
    // estimates.
    {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_merge_cgus");
        merge_codegen_units(cx, &mut codegen_units);
        debug_dump(tcx, "MERGE", &codegen_units);
    }

    // Make as many symbols "internal" as possible, so LLVM has more freedom to
    // optimize.
    if !tcx.sess.link_dead_code() {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_internalize_symbols");
        internalize_symbols(cx, &mut codegen_units, internalization_candidates);

        debug_dump(tcx, "INTERNALIZE", &codegen_units);
    }

    // Mark one CGU for dead code, if necessary.
    if tcx.sess.instrument_coverage() {
        mark_code_coverage_dead_code_cgu(&mut codegen_units);
    }

    // Ensure CGUs are sorted by name, so that we get deterministic results.
    if !codegen_units.is_sorted_by(|a, b| a.name().as_str() <= b.name().as_str()) {
        let mut names = String::new();
        for cgu in codegen_units.iter() {
            names += &format!("- {}\n", cgu.name());
        }
        bug!("unsorted CGUs:\n{names}");
    }

    codegen_units
}

fn place_mono_items<'tcx, I>(cx: &PartitioningCx<'_, 'tcx>, mono_items: I) -> PlacedMonoItems<'tcx>
where
    I: Iterator<Item = MonoItem<'tcx>>,
{
    let mut codegen_units = UnordMap::default();
    let is_incremental_build = cx.tcx.sess.opts.incremental.is_some();
    let mut internalization_candidates = UnordSet::default();

    // Determine if monomorphizations instantiated in this crate will be made
    // available to downstream crates. This depends on whether we are in
    // share-generics mode and whether the current crate can even have
    // downstream crates.
    let can_export_generics = cx.tcx.local_crate_exports_generics();
    let always_export_generics = can_export_generics && cx.tcx.sess.opts.share_generics();

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(cx.tcx);
    let cgu_name_cache = &mut UnordMap::default();

    for mono_item in mono_items {
        // Handle only root (GloballyShared) items directly here. Inlined (LocalCopy) items
        // are handled at the bottom of the loop based on reachability, with one exception.
        // The #[lang = "start"] item is the program entrypoint, so there are no calls to it in MIR.
        // So even if its mode is LocalCopy, we need to treat it like a root.
        match mono_item.instantiation_mode(cx.tcx) {
            InstantiationMode::GloballyShared { .. } => {}
            InstantiationMode::LocalCopy => {
                // Items added after the main mono collection pass (for example,
                // trait-cast vtable methods discovered while resolving table
                // allocations) have no usage-map edges. Treat those orphaned
                // LocalCopy items as synthetic roots so they still get placed
                // into a CGU and emitted for codegen.
                if cx.usage_map.used_map.contains_key(&mono_item)
                    || !cx.usage_map.get_user_items(mono_item).is_empty()
                {
                    continue;
                }
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
            can_export_generics,
            always_export_generics,
        );

        if visibility == Visibility::Hidden && can_be_internalized {
            internalization_candidates.insert(mono_item);
        }
        let size_estimate = mono_item.size_estimate(cx.tcx);

        cgu.items_mut()
            .insert(mono_item, MonoItemData { inlined: false, linkage, visibility, size_estimate });

        // Get all inlined items that are reachable from `mono_item` without
        // going via another root item. This includes drop-glue, functions from
        // external crates, and local functions the definition of which is
        // marked with `#[inline]`.
        let mut reachable_inlined_items = FxIndexSet::default();
        get_reachable_inlined_items(cx.tcx, mono_item, cx.usage_map, &mut reachable_inlined_items);

        // Add those inlined items. It's possible an inlined item is reachable
        // from multiple root items within a CGU, which is fine, it just means
        // the `insert` will be a no-op.
        for inlined_item in reachable_inlined_items {
            // Trait-cast delayed instances must never be CGU-private: the
            // matching symbol from an upstream dylib (built with the
            // instantiating-crate suffix stripped from the mangled name) is
            // resolved against the global crate's DYNSYM at runtime. A
            // CGU-private (Internal) copy wouldn't appear in DYNSYM at all,
            // leaving the dylib's reloc unresolved. Promote to
            // External + Protected instead.
            let delayed_inlined = matches!(inlined_item, MonoItem::Fn(i)
                if cx.tcx.is_global_crate()
                    && cx.tcx.is_transitively_delayed_instance(i));
            let (ilinkage, ivisibility) = if delayed_inlined {
                (Linkage::External, Visibility::Protected)
            } else {
                (Linkage::Internal, Visibility::Default)
            };
            cgu.items_mut().entry(inlined_item).or_insert_with(|| MonoItemData {
                inlined: true,
                linkage: ilinkage,
                visibility: ivisibility,
                size_estimate: inlined_item.size_estimate(cx.tcx),
            });
        }
    }

    // Always ensure we have at least one CGU; otherwise, if we have a
    // crate with just types (for example), we could wind up with no CGU.
    if codegen_units.is_empty() {
        let cgu_name = fallback_cgu_name(cgu_name_builder);
        codegen_units.insert(cgu_name, CodegenUnit::new(cgu_name));
    }

    let mut codegen_units: Vec<_> = cx.tcx.with_stable_hashing_context(|mut hcx| {
        codegen_units.into_items().map(|(_, cgu)| cgu).collect_sorted(&mut hcx, true)
    });

    for cgu in codegen_units.iter_mut() {
        cgu.compute_size_estimate();
    }

    return PlacedMonoItems { codegen_units, internalization_candidates };

    fn get_reachable_inlined_items<'tcx>(
        tcx: TyCtxt<'tcx>,
        item: MonoItem<'tcx>,
        usage_map: &UsageMap<'tcx>,
        visited: &mut FxIndexSet<MonoItem<'tcx>>,
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
    assert!(codegen_units.is_sorted_by(|a, b| a.name().as_str() <= b.name().as_str()));

    // This map keeps track of what got merged into what.
    let mut cgu_contents: UnordMap<Symbol, Vec<Symbol>> =
        codegen_units.iter().map(|cgu| (cgu.name(), vec![cgu.name()])).collect();

    // If N is the maximum number of CGUs, and the CGUs are sorted from largest
    // to smallest, we repeatedly find which CGU in codegen_units[N..] has the
    // greatest overlap of inlined items with codegen_units[N-1], merge that
    // CGU into codegen_units[N-1], then re-sort by size and repeat.
    //
    // We use inlined item overlap to guide this merging because it minimizes
    // duplication of inlined items, which makes LLVM be faster and generate
    // better and smaller machine code.
    //
    // Why merge into codegen_units[N-1]? We want CGUs to have similar sizes,
    // which means we don't want codegen_units[0..N] (the already big ones)
    // getting any bigger, if we can avoid it. When we have more than N CGUs
    // then at least one of the biggest N will have to grow. codegen_units[N-1]
    // is the smallest of those, and so has the most room to grow.
    let max_codegen_units = cx.tcx.sess.codegen_units().as_usize();
    while codegen_units.len() > max_codegen_units {
        // Sort small CGUs to the back.
        codegen_units.sort_by_key(|cgu| cmp::Reverse(cgu.size_estimate()));

        let cgu_dst = &codegen_units[max_codegen_units - 1];

        // Find the CGU that overlaps the most with `cgu_dst`. In the case of a
        // tie, favour the earlier (bigger) CGU.
        let mut max_overlap = 0;
        let mut max_overlap_i = max_codegen_units;
        for (i, cgu_src) in codegen_units.iter().enumerate().skip(max_codegen_units) {
            if cgu_src.size_estimate() <= max_overlap {
                // None of the remaining overlaps can exceed `max_overlap`, so
                // stop looking.
                break;
            }

            let overlap = compute_inlined_overlap(cgu_dst, cgu_src);
            if overlap > max_overlap {
                max_overlap = overlap;
                max_overlap_i = i;
            }
        }

        let mut cgu_src = codegen_units.swap_remove(max_overlap_i);
        let cgu_dst = &mut codegen_units[max_codegen_units - 1];

        // Move the items from `cgu_src` to `cgu_dst`. Some of them may be
        // duplicate inlined items, in which case the destination CGU is
        // unaffected. Recalculate size estimates afterwards.
        cgu_dst.items_mut().append(cgu_src.items_mut());
        cgu_dst.compute_size_estimate();

        // Record that `cgu_dst` now contains all the stuff that was in
        // `cgu_src` before.
        let mut consumed_cgu_names = cgu_contents.remove(&cgu_src.name()).unwrap();
        cgu_contents.get_mut(&cgu_dst.name()).unwrap().append(&mut consumed_cgu_names);
    }

    // Having multiple CGUs can drastically speed up compilation. But for
    // non-incremental builds, tiny CGUs slow down compilation *and* result in
    // worse generated code. So we don't allow CGUs smaller than this (unless
    // there is just one CGU, of course). Note that CGU sizes of 100,000+ are
    // common in larger programs, so this isn't all that large.
    const NON_INCR_MIN_CGU_SIZE: usize = 1800;

    // Repeatedly merge the two smallest codegen units as long as: it's a
    // non-incremental build, and the user didn't specify a CGU count, and
    // there are multiple CGUs, and some are below the minimum size.
    //
    // The "didn't specify a CGU count" condition is because when an explicit
    // count is requested we observe it as closely as possible. For example,
    // the `compiler_builtins` crate sets `codegen-units = 10000` and it's
    // critical they aren't merged. Also, some tests use explicit small values
    // and likewise won't work if small CGUs are merged.
    while cx.tcx.sess.opts.incremental.is_none()
        && matches!(cx.tcx.sess.codegen_units(), CodegenUnits::Default(_))
        && codegen_units.len() > 1
        && codegen_units.iter().any(|cgu| cgu.size_estimate() < NON_INCR_MIN_CGU_SIZE)
    {
        // Sort small cgus to the back.
        codegen_units.sort_by_key(|cgu| cmp::Reverse(cgu.size_estimate()));

        let mut smallest = codegen_units.pop().unwrap();
        let second_smallest = codegen_units.last_mut().unwrap();

        // Move the items from `smallest` to `second_smallest`. Some of them
        // may be duplicate inlined items, in which case the destination CGU is
        // unaffected. Recalculate size estimates afterwards.
        second_smallest.items_mut().append(smallest.items_mut());
        second_smallest.compute_size_estimate();

        // Don't update `cgu_contents`, that's only for incremental builds.
    }

    let cgu_name_builder = &mut CodegenUnitNameBuilder::new(cx.tcx);

    // Rename the newly merged CGUs.
    if cx.tcx.sess.opts.incremental.is_some() {
        // If we are doing incremental compilation, we want CGU names to
        // reflect the path of the source level module they correspond to.
        // For CGUs that contain the code of multiple modules because of the
        // merging done above, we use a concatenation of the names of all
        // contained CGUs.
        let new_cgu_names = UnordMap::from(
            cgu_contents
                .items()
                // This `filter` makes sure we only update the name of CGUs that
                // were actually modified by merging.
                .filter(|(_, cgu_contents)| cgu_contents.len() > 1)
                .map(|(current_cgu_name, cgu_contents)| {
                    let mut cgu_contents: Vec<&str> =
                        cgu_contents.iter().map(|s| s.as_str()).collect();

                    // Sort the names, so things are deterministic and easy to
                    // predict. We are sorting primitive `&str`s here so we can
                    // use unstable sort.
                    cgu_contents.sort_unstable();

                    (*current_cgu_name, cgu_contents.join("--"))
                }),
        );

        for cgu in codegen_units.iter_mut() {
            if let Some(new_cgu_name) = new_cgu_names.get(&cgu.name()) {
                let new_cgu_name = if cx.tcx.sess.opts.unstable_opts.human_readable_cgu_names {
                    Symbol::intern(&CodegenUnit::shorten_name(new_cgu_name))
                } else {
                    // If we don't require CGU names to be human-readable,
                    // we use a fixed length hash of the composite CGU name
                    // instead.
                    Symbol::intern(&CodegenUnit::mangle_name(new_cgu_name))
                };
                cgu.set_name(new_cgu_name);
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
        // - [1,9]     CGUs: `0`, `1`, `2`, ...
        // - [10,99]   CGUs: `00`, `01`, `02`, ...
        // - [100,999] CGUs: `000`, `001`, `002`, ...
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

/// Compute the combined size of all inlined items that appear in both `cgu1`
/// and `cgu2`.
fn compute_inlined_overlap<'tcx>(cgu1: &CodegenUnit<'tcx>, cgu2: &CodegenUnit<'tcx>) -> usize {
    // Either order works. We pick the one that involves iterating over fewer
    // items.
    let (src_cgu, dst_cgu) =
        if cgu1.items().len() <= cgu2.items().len() { (cgu1, cgu2) } else { (cgu2, cgu1) };

    let mut overlap = 0;
    for (item, data) in src_cgu.items().iter() {
        if data.inlined && dst_cgu.items().contains_key(item) {
            overlap += data.size_estimate;
        }
    }
    overlap
}

fn internalize_symbols<'tcx>(
    cx: &PartitioningCx<'_, 'tcx>,
    codegen_units: &mut [CodegenUnit<'tcx>],
    internalization_candidates: UnordSet<MonoItem<'tcx>>,
) {
    /// For symbol internalization, we need to know whether a symbol/mono-item
    /// is used from outside the codegen unit it is defined in. This type is
    /// used to keep track of that.
    #[derive(Clone, PartialEq, Eq, Debug)]
    enum MonoItemPlacement {
        SingleCgu(Symbol),
        MultipleCgus,
    }

    let mut mono_item_placements = UnordMap::default();
    let single_codegen_unit = codegen_units.len() == 1;

    if !single_codegen_unit {
        for cgu in codegen_units.iter() {
            for item in cgu.items().keys() {
                // If there is more than one codegen unit, we need to keep track
                // in which codegen units each monomorphization is placed.
                match mono_item_placements.entry(*item) {
                    Entry::Occupied(e) => {
                        let placement = e.into_mut();
                        debug_assert!(match *placement {
                            MonoItemPlacement::SingleCgu(cgu_name) => cgu_name != cgu.name(),
                            MonoItemPlacement::MultipleCgus => true,
                        });
                        *placement = MonoItemPlacement::MultipleCgus;
                    }
                    Entry::Vacant(e) => {
                        e.insert(MonoItemPlacement::SingleCgu(cgu.name()));
                    }
                }
            }
        }
    }

    // For each internalization candidates in each codegen unit, check if it is
    // used from outside its defining codegen unit.
    for cgu in codegen_units {
        let home_cgu = MonoItemPlacement::SingleCgu(cgu.name());

        for (item, data) in cgu.items_mut() {
            if !internalization_candidates.contains(item) {
                // This item is no candidate for internalizing, so skip it.
                continue;
            }

            if !single_codegen_unit {
                debug_assert_eq!(mono_item_placements[item], home_cgu);

                if cx
                    .usage_map
                    .get_user_items(*item)
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

            // When LTO inlines the caller of a naked function, it will attempt but fail to make the
            // naked function symbol visible. To ensure that LTO works correctly, do not default
            // naked functions to internal linkage and default visibility.
            if let MonoItem::Fn(instance) = item {
                let flags = cx.tcx.codegen_instance_attrs(instance.def).flags;
                if flags.contains(CodegenFnAttrFlags::NAKED) {
                    continue;
                }
            }

            // If we got here, we did not find any uses from other CGUs, so
            // it's fine to make this monomorphization internal.
            data.linkage = Linkage::Internal;
            data.visibility = Visibility::Default;
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
        .filter(|cgu| cgu.items().iter().any(|(_, data)| data.linkage == Linkage::External))
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
                ty::InstanceKind::Item(def) => def,
                ty::InstanceKind::VTableShim(..)
                | ty::InstanceKind::ReifyShim(..)
                | ty::InstanceKind::FnPtrShim(..)
                | ty::InstanceKind::ClosureOnceShim { .. }
                | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
                | ty::InstanceKind::Intrinsic(..)
                | ty::InstanceKind::DropGlue(..)
                | ty::InstanceKind::Virtual(..)
                | ty::InstanceKind::CloneShim(..)
                | ty::InstanceKind::ThreadLocalShim(..)
                | ty::InstanceKind::FnPtrAddrShim(..)
                | ty::InstanceKind::FutureDropPollShim(..)
                | ty::InstanceKind::AsyncDropGlue(..)
                | ty::InstanceKind::AsyncDropGlueCtorShim(..) => return None,
            };

            // If this is a method, we want to put it into the same module as
            // its self-type. If the self-type does not provide a characteristic
            // DefId, we use the location of the impl after all.

            let assoc_parent = tcx.assoc_parent(def_id);

            if let Some((_, DefKind::Trait)) = assoc_parent {
                let self_ty = instance.args.type_at(0);
                // This is a default implementation of a trait method.
                return characteristic_def_id_of_type(self_ty).or(Some(def_id));
            }

            if let Some((impl_def_id, DefKind::Impl { of_trait })) = assoc_parent {
                if of_trait
                    && tcx.sess.opts.incremental.is_some()
                    && tcx.is_lang_item(tcx.impl_trait_id(impl_def_id), LangItem::Drop)
                {
                    // Put `Drop::drop` into the same cgu as `drop_in_place`
                    // since `drop_in_place` is the only thing that can
                    // call it.
                    return None;
                }

                // This is a method within an impl, find out what the self-type is:
                let impl_self_ty = tcx.instantiate_and_normalize_erasing_regions(
                    instance.args,
                    ty::TypingEnv::fully_monomorphized(),
                    tcx.type_of(impl_def_id),
                );
                if let Some(def_id) = characteristic_def_id_of_type(impl_self_ty) {
                    return Some(def_id);
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
    can_export_generics: bool,
    always_export_generics: bool,
) -> (Linkage, Visibility) {
    if let Some(explicit_linkage) = mono_item.explicit_linkage(tcx) {
        return (explicit_linkage, Visibility::Default);
    }
    let vis = mono_item_visibility(
        tcx,
        mono_item,
        can_be_internalized,
        can_export_generics,
        always_export_generics,
    );
    (Linkage::External, vis)
}

type CguNameCache = UnordMap<(DefId, bool), Symbol>;

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
    can_export_generics: bool,
    always_export_generics: bool,
) -> Visibility {
    // Trait-cast delayed instances codegen'd by the global crate must be
    // `Protected`: the symbol needs to appear in `DT_DYNSYM` so upstream
    // dylibs loaded alongside can resolve their vtable relocs at runtime,
    // but we must **not** let another global crate loaded in the same
    // process interpose our local intrinsic calls (the AllocId-rejection
    // check is predicated on local calls using local bodies).
    // `Visibility::Protected` maps to ELF `STV_PROTECTED` ("hide upward,
    // export downward"); on object formats that don't support it,
    // degrades to `Default`, which is still correct for the common
    // single-global-crate case.
    if let MonoItem::Fn(instance) = mono_item
        && tcx.is_global_crate()
        && tcx.is_transitively_delayed_instance(*instance)
    {
        *can_be_internalized = false;
        return Visibility::Protected;
    }

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
        InstanceKind::Item(def_id)
        | InstanceKind::DropGlue(def_id, Some(_))
        | InstanceKind::FutureDropPollShim(def_id, _, _)
        | InstanceKind::AsyncDropGlue(def_id, _)
        | InstanceKind::AsyncDropGlueCtorShim(def_id, _) => def_id,

        // We match the visibility of statics here
        InstanceKind::ThreadLocalShim(def_id) => {
            return static_visibility(tcx, can_be_internalized, def_id);
        }

        // These are all compiler glue and such, never exported, always hidden.
        InstanceKind::VTableShim(..)
        | InstanceKind::ReifyShim(..)
        | InstanceKind::FnPtrShim(..)
        | InstanceKind::Virtual(..)
        | InstanceKind::Intrinsic(..)
        | InstanceKind::ClosureOnceShim { .. }
        | InstanceKind::ConstructCoroutineInClosureShim { .. }
        | InstanceKind::DropGlue(..)
        | InstanceKind::CloneShim(..)
        | InstanceKind::FnPtrAddrShim(..) => return Visibility::Hidden,
    };

    // Both the `start_fn` lang item and `main` itself should not be exported,
    // so we give them with `Hidden` visibility but these symbols are
    // only referenced from the actual `main` symbol which we unfortunately
    // don't know anything about during partitioning/collection. As a result we
    // forcibly keep this symbol out of the `internalization_candidates` set.
    //
    // FIXME: eventually we don't want to always force this symbol to have
    //        hidden visibility, it should indeed be a candidate for
    //        internalization, but we have to understand that it's referenced
    //        from the `main` symbol we'll generate later.
    //
    //        This may be fixable with a new `InstanceKind` perhaps? Unsure!
    if tcx.is_entrypoint(def_id) {
        *can_be_internalized = false;
        return Visibility::Hidden;
    }

    let is_generic = instance.args.non_erasable_generics().next().is_some();

    // Upstream `DefId` instances get different handling than local ones.
    let Some(def_id) = def_id.as_local() else {
        return if is_generic
            && (always_export_generics
                || (can_export_generics
                    && tcx.codegen_fn_attrs(def_id).inline == InlineAttr::Never))
        {
            // If it is an upstream monomorphization and we export generics, we must make
            // it available to downstream crates.
            *can_be_internalized = false;
            default_visibility(tcx, def_id, true)
        } else {
            Visibility::Hidden
        };
    };

    if is_generic {
        if always_export_generics
            || (can_export_generics && tcx.codegen_fn_attrs(def_id).inline == InlineAttr::Never)
        {
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
        // There's three categories of items here:
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
        //
        // * Externally implementable items. They work (in this case) pretty much the same as
        //   RUSTC_STD_INTERNAL_SYMBOL in that their implementation is also chosen later in
        //   the compilation process and we can't let them be internalized and they can't
        //   show up as an external interface.
        let attrs = tcx.codegen_fn_attrs(def_id);
        if attrs.flags.intersects(
            CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL
                | CodegenFnAttrFlags::EXTERNALLY_IMPLEMENTABLE_ITEM,
        ) {
            *can_be_internalized = false;
        }

        Visibility::Hidden
    }
}

fn default_visibility(tcx: TyCtxt<'_>, id: DefId, is_generic: bool) -> Visibility {
    // Fast-path to avoid expensive query call below
    if tcx.sess.default_visibility() == SymbolVisibility::Interposable {
        return Visibility::Default;
    }

    let export_level = if is_generic {
        // Generic functions never have export-level C.
        SymbolExportLevel::Rust
    } else {
        match tcx.reachable_non_generics(id.krate).get(&id) {
            Some(SymbolExportInfo { level: SymbolExportLevel::C, .. }) => SymbolExportLevel::C,
            _ => SymbolExportLevel::Rust,
        }
    };

    match export_level {
        // C-export level items remain at `Default` to allow C code to
        // access and interpose them.
        SymbolExportLevel::C => Visibility::Default,

        // For all other symbols, `default_visibility` determines which visibility to use.
        SymbolExportLevel::Rust => tcx.sess.default_visibility().into(),
    }
}

fn debug_dump<'a, 'tcx: 'a>(tcx: TyCtxt<'tcx>, label: &str, cgus: &[CodegenUnit<'tcx>]) {
    let dump = move || {
        use std::fmt::Write;

        let mut num_cgus = 0;
        let mut all_cgu_sizes = Vec::new();

        // Note: every unique root item is placed exactly once, so the number
        // of unique root items always equals the number of placed root items.
        //
        // Also, unreached inlined items won't be counted here. This is fine.

        let mut inlined_items = UnordSet::default();

        let mut root_items = 0;
        let mut unique_inlined_items = 0;
        let mut placed_inlined_items = 0;

        let mut root_size = 0;
        let mut unique_inlined_size = 0;
        let mut placed_inlined_size = 0;

        for cgu in cgus.iter() {
            num_cgus += 1;
            all_cgu_sizes.push(cgu.size_estimate());

            for (item, data) in cgu.items() {
                if !data.inlined {
                    root_items += 1;
                    root_size += data.size_estimate;
                } else {
                    if inlined_items.insert(item) {
                        unique_inlined_items += 1;
                        unique_inlined_size += data.size_estimate;
                    }
                    placed_inlined_items += 1;
                    placed_inlined_size += data.size_estimate;
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
                cgu.items().values().map(|data| data.size_estimate).collect();
            placed_item_sizes.sort_unstable_by_key(|&n| cmp::Reverse(n));
            let sizes = list(&placed_item_sizes);

            let _ = writeln!(s, "- CGU[{i}]");
            let _ = writeln!(s, "  - {name}, size: {size}");
            let _ =
                writeln!(s, "  - items: {num_items}, mean size: {mean_size:.1}, sizes: {sizes}",);

            for (item, data) in cgu.items_in_deterministic_order(tcx) {
                let linkage = data.linkage;
                let symbol_name = item.symbol_name(tcx).name;
                let symbol_hash_start = symbol_name.rfind('h');
                let symbol_hash = symbol_hash_start.map_or("<no hash>", |i| &symbol_name[i..]);
                let kind = if !data.inlined { "root" } else { "inlined" };
                let size = data.size_estimate;
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

            format!("[{}]", v.join(", "))
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

            tcx.dcx().emit_fatal(SymbolAlreadyDefined { span, symbol: sym1.to_string() });
        }
    }
}

/// Emit the `unused_cast_target` lint for every `trait_metadata_index`
/// request whose sub_trait has no satisfying concrete type in the final
/// binary's trait graph.
///
/// This is the final crate of compilation (binary / staticlib / cdylib),
/// so the set of concrete types implementing the root is known and we can
/// tell whether any of them also implement the sub_trait. If none do, the
/// cast will always return `Err` at runtime.
///
/// Span recovery: each request carries the intrinsic `Instance`; we walk
/// the crate graph's delayed requests once to map intrinsic → caller and
/// use the caller's `def_span` as the lint's primary span. Cross-crate
/// casts land on the caller's foreign def_span; local casts land near
/// the `cast!` invocation.
fn emit_unused_cast_target_lint<'tcx>(tcx: TyCtxt<'tcx>, requests: &TraitCastRequests<'tcx>) {
    use std::iter;

    use rustc_hir::CRATE_HIR_ID;
    use rustc_lint_defs::builtin::UNUSED_CAST_TARGET;
    use rustc_middle::ty::Instance;
    use rustc_span::DUMMY_SP;

    use crate::errors::UnusedCastTargetLint;
    use crate::trait_graph::resolve_dyn_satisfaction;

    if requests.index_requests.is_empty() {
        return;
    }

    // Map each intrinsic Instance to the def_span of the caller that references it.
    let mut intrinsic_caller_span: FxHashMap<Instance<'tcx>, rustc_span::Span> =
        FxHashMap::default();
    for &cnum in iter::once(&LOCAL_CRATE).chain(tcx.crates(())) {
        for delayed in tcx.delayed_codegen_requests(cnum) {
            for &intrinsic in delayed.intrinsic_callees {
                intrinsic_caller_span
                    .entry(intrinsic)
                    .or_insert_with(|| tcx.def_span(delayed.instance.def_id()));
            }
        }
    }

    #[allow(rustc::potential_query_instability)]
    for req in &requests.index_requests {
        let graph = tcx.trait_cast_graph(req.super_trait);
        let any_satisfies = graph
            .concrete_types
            .items()
            .any(|ct| resolve_dyn_satisfaction(tcx, **ct, req.sub_trait).is_some());
        if any_satisfies {
            continue;
        }
        let span = intrinsic_caller_span.get(&req.instance).copied().unwrap_or(DUMMY_SP);
        tcx.emit_node_span_lint(
            UNUSED_CAST_TARGET,
            CRATE_HIR_ID,
            span,
            UnusedCastTargetLint { span, root: req.super_trait, target: req.sub_trait },
        );
    }
}

/// Called from within `collect_and_partition_mono_items`, after mono
/// collection completes but before partitioning. Resolves all
/// delayed codegen requests into `MonoItem::Fn` entries that are
/// inserted into `mono_items` before partitioning distributes
/// items into codegen units.
/// Only runs in a global crate (binary, staticlib, cdylib).
fn resolve_trait_cast_globals<'tcx>(tcx: TyCtxt<'tcx>, mono_items: &mut Cow<'_, [MonoItem<'tcx>]>) {
    if !tcx.is_global_crate() {
        return; // Non-global crates defer to the global crate.
    }

    let requests = tcx.gather_trait_cast_requests(());
    if requests.is_empty() {
        return; // No trait casting in the entire program.
    }

    // Build the intrinsic resolution lookup table. Query results
    // (trait_cast_layout, trait_cast_table, trait_cast_table_alloc)
    // are driven on-demand within build_intrinsic_resolutions and
    // cached by the dep graph for incremental reuse.
    let resolutions = crate::table_layout::build_intrinsic_resolutions(tcx, &requests);

    // Fire `unused_cast_target` lint for every `trait_metadata_index`
    // request whose sub_trait has no concrete-type implementer in the
    // final binary. Such casts always return `Err` at runtime.
    emit_unused_cast_target_lint(tcx, &requests);

    // Cascading canonicalization: process all caller DelayedInstances
    // (directly + transitively sensitive) bottom-up, resolving
    // intrinsics, rewriting callee references through the condensation
    // map, patching MIR, feeding codegen_mir, and inserting
    // MonoItem::Fn entries into mono_items.
    //
    // Pulls DelayedInstances from delayed_codegen_requests directly —
    // independent of TraitCastRequests.
    cascade_canonicalize(tcx, &resolutions, mono_items);

    // Collect mono items for vtable methods referenced by trait cast
    // tables. For each (super_trait, concrete_type) table, iterate the
    // sub-traits and collect vtable methods for implemented sub-traits.
    // Uses create_mono_items_for_vtable_methods (same path as the normal
    // collector) to ensure Instance consistency. Dedup against items
    // already collected from direct unsizing points.
    collect_trait_cast_vtable_methods(tcx, &requests, mono_items);
    collect_trait_cast_table_backing_items(tcx, &resolutions, mono_items);
}

/// Collect vtable method mono items for all (concrete_type, sub_trait) pairs
/// in the trait cast tables. These vtables are generated during resolution
/// but their methods must be added as mono items for codegen.
///
/// Deduplicates against already-collected items (from direct unsizing casts)
/// and across sub-trait vtables (which share supertrait methods).
fn collect_trait_cast_vtable_methods<'tcx>(
    tcx: TyCtxt<'tcx>,
    requests: &TraitCastRequests<'tcx>,
    mono_items: &mut Cow<'_, [MonoItem<'tcx>]>,
) {
    use crate::trait_graph::resolve_dyn_satisfaction;

    // Deduplicate (super_trait, concrete_type) pairs across requests.
    let table_pairs: FxHashSet<(Ty<'tcx>, Ty<'tcx>)> =
        requests.table_requests.iter().map(|r| (r.super_trait, r.concrete_type)).collect();

    if table_pairs.is_empty() {
        return;
    }

    let mut seen: FxHashSet<MonoItem<'tcx>> = mono_items.iter().copied().collect();
    let mut new_items = Vec::new();
    let mut candidate_items = Vec::new();

    // Iteration order is irrelevant — we are collecting into a dedup set.
    #[allow(rustc::potential_query_instability)]
    for &(super_trait, concrete_type) in &table_pairs {
        let layout = tcx.trait_cast_layout(super_trait);
        for sub_trait in layout.sub_traits() {
            if resolve_dyn_satisfaction(tcx, concrete_type, sub_trait).is_none() {
                continue;
            }
            crate::collector::collect_vtable_methods_for_trait_cast(
                tcx,
                sub_trait,
                concrete_type,
                &mut candidate_items,
            );
            for item in candidate_items.drain(..) {
                if seen.insert(item) {
                    new_items.push(item);
                }
            }
        }
    }

    if !new_items.is_empty() {
        mono_items.to_mut().extend(new_items);
    }
}

/// Collect mono items reachable from the actual trait-cast table allocations.
///
/// This is a conservative backstop for cases where reconstructing the
/// (sub-trait, concrete-type) pairs misses an item that the emitted table
/// or its referenced vtables nevertheless contain.
fn collect_trait_cast_table_backing_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    resolutions: &IntrinsicResolutions<'tcx>,
    mono_items: &mut Cow<'_, [MonoItem<'tcx>]>,
) {
    if resolutions.table_alloc_ids.is_empty() {
        return;
    }

    let mut seen: FxHashSet<MonoItem<'tcx>> = mono_items.iter().copied().collect();
    let mut new_items = Vec::new();
    let mut candidate_items = Vec::new();

    for &alloc_id in &resolutions.table_alloc_ids {
        crate::collector::collect_alloc_items_for_trait_cast(tcx, alloc_id, &mut candidate_items);
        for item in candidate_items.drain(..) {
            if seen.insert(item) {
                new_items.push(item);
            }
        }
    }

    if !new_items.is_empty() {
        mono_items.to_mut().extend(new_items);
    }
}

/// Process all delayed codegen Instances bottom-up through the sensitive
/// sub-graph: apply callee substitutions, resolve intrinsic calls,
/// canonicalize condensed Instances via trampoline bodies, feed patched
/// MIR via `codegen_mir`, and insert `MonoItem::Fn` entries into
/// `mono_items` for partitioning.
///
/// Condensation-based deduplication: two Instances that belong to the
/// same condensation group (identical admissibility vectors across all
/// concrete types) produce identical resolved bodies. The canonical
/// Instance (smallest `OutlivesClass` under `StableOrd`) receives the
/// full resolved body; non-canonical Instances receive a trampoline body
/// that tail-calls the canonical.
///
/// Deduplication cascades through the bottom-up traversal: when leaf
/// Instances condense, their callers' patched MIR references the same
/// canonical callee. Callers that differ only in which condensed callee
/// they reference now produce identical patched bodies and are
/// themselves deduplicated.
fn cascade_canonicalize<'tcx>(
    tcx: TyCtxt<'tcx>,
    resolutions: &IntrinsicResolutions<'tcx>,
    mono_items: &mut Cow<'_, [MonoItem<'tcx>]>,
) {
    use std::collections::BTreeMap;

    use rustc_data_structures::graph::scc::Sccs;
    use rustc_data_structures::graph::vec_graph::VecGraph;
    use rustc_middle::mono::DelayedInstance;
    use rustc_middle::ty::trait_cast::OutlivesClass;

    let dump = tcx.sess.opts.unstable_opts.dump_trait_cast_canonicalization;

    // Collect all delayed Instances from all crates, deduplicating by Instance.
    let mut all_delayed: Vec<&DelayedInstance<'tcx>> = Vec::new();
    let mut seen: FxHashSet<ty::Instance<'tcx>> = FxHashSet::default();
    for delayed in tcx.delayed_codegen_requests(LOCAL_CRATE) {
        if seen.insert(delayed.instance) {
            all_delayed.push(delayed);
        }
    }
    for &cnum in tcx.crates(()) {
        for delayed in tcx.delayed_codegen_requests(cnum) {
            if seen.insert(delayed.instance) {
                all_delayed.push(delayed);
            }
        }
    }

    if all_delayed.is_empty() {
        if dump {
            eprintln!("=== Trait-Cast Canonicalization ===");
            eprintln!("  Total delayed instances: 0");
            eprintln!("  Depth levels: 0");
            eprintln!("  Canon map summary:");
            eprintln!("    total redirections: 0");
        }
        return;
    }

    // Build the delayed-Instance dependency graph and compute depths
    // via VecGraph + Sccs. The dependency graph is a DAG (the call
    // graph is acyclic after augmentation), so every SCC is a
    // singleton — Sccs gives us a topological ordering for free,
    // and depth is computed in a single O(V+E) pass.
    rustc_index::newtype_index! {
        #[orderable]
        struct DelayIdx {}
    }
    rustc_index::newtype_index! {
        #[orderable]
        struct DelaySccIdx {}
    }

    let instance_to_idx: FxHashMap<ty::Instance<'tcx>, usize> =
        all_delayed.iter().enumerate().map(|(i, d)| (d.instance, i)).collect();

    let mut edge_pairs: Vec<(DelayIdx, DelayIdx)> = Vec::new();
    for (i, d) in all_delayed.iter().enumerate() {
        for &(_, callee) in d.callee_substitutions {
            if let Some(&j) = instance_to_idx.get(&callee) {
                edge_pairs.push((DelayIdx::from(i), DelayIdx::from(j)));
            }
        }
    }

    let delay_graph = VecGraph::<DelayIdx>::new(all_delayed.len(), edge_pairs);
    let delay_sccs = Sccs::<DelayIdx, DelaySccIdx>::new(&delay_graph);

    // Single-pass depth computation in dependency order (O(V+E)).
    // all_sccs() visits callees before callers, so successor depths
    // are already resolved when we compute a node's depth.
    let mut scc_depth: Vec<usize> = vec![0; delay_sccs.num_sccs()];
    for scc in delay_sccs.all_sccs() {
        let d = delay_sccs
            .successors(scc)
            .iter()
            .copied()
            .map(|succ| scc_depth[succ.index()] + 1)
            .max()
            .unwrap_or(0);
        scc_depth[scc.index()] = d;
    }
    let depth: Vec<usize> = (0..all_delayed.len())
        .map(|i| scc_depth[delay_sccs.scc(DelayIdx::from(i)).index()])
        .collect();

    // --- Seed canon_map from condensation groups (leaf level) ---
    //
    // For each root trait's layout, the condensation groups identify
    // which (sub_trait, outlives_class) pairs share the same table slot.
    // Group the directly-sensitive delayed Instances by
    // (def_id, base_args) — same function, same type args, differing
    // only in Outlives entries. Within each group, Instances whose
    // OutlivesClass maps to the same condensation slot are equivalent.
    // Pick the canonical (smallest OutlivesClass under StableOrd),
    // map the rest → canonical.
    let mut canon_map: FxHashMap<ty::Instance<'tcx>, ty::Instance<'tcx>> = FxHashMap::default();

    seed_canon_map_from_condensation(tcx, &all_delayed, &mut canon_map);

    // Group Instances by depth level for bottom-up processing.
    // BTreeMap gives us ascending depth order (leaves first).
    let mut depth_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..all_delayed.len() {
        depth_groups.entry(depth[i]).or_default().push(i);
    }

    if dump {
        eprintln!("=== Trait-Cast Canonicalization ===");
        eprintln!("  Total delayed instances: {}", all_delayed.len());
        eprintln!("  Depth levels: {}", depth_groups.len());
    }

    let mut new_mono_items: Vec<MonoItem<'tcx>> = Vec::new();

    for (d, indices) in &depth_groups {
        // Order indices deterministically for dump output by stable
        // fingerprint of the Instance. This does not affect observable
        // behavior; it only reorders the emission.
        let dump_order: Vec<usize> = if dump {
            let mut v = indices.clone();
            tcx.with_stable_hashing_context(|mut hcx| {
                v.sort_by_cached_key(|&idx| {
                    use rustc_data_structures::fingerprint::Fingerprint;
                    use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
                    let mut hasher = StableHasher::new();
                    all_delayed[idx].instance.hash_stable(&mut hcx, &mut hasher);
                    hasher.finish::<Fingerprint>()
                });
            });
            eprintln!("  Depth {d}: {} instance(s)", indices.len());
            eprintln!("    Phase 1 (patch):");
            v
        } else {
            Vec::new()
        };
        // --- Phase 1: Patch and resolve all canonical Instances at this depth ---
        //
        // For canonical Instances (not in canon_map): clone base MIR,
        // apply callee substitutions through canon_map, resolve
        // intrinsic calls, and record the patched body.
        //
        // For non-canonical Instances (already in canon_map from
        // leaf-level seeding or prior depth levels): skip patching,
        // a trampoline body will be generated in Phase 3.
        let mut patched_bodies: FxHashMap<ty::Instance<'tcx>, &'tcx rustc_middle::mir::Body<'tcx>> =
            FxHashMap::default();

        // Iteration order for the patching logic itself is irrelevant
        // (the loop only mutates the per-instance `patched_bodies`
        // entry). Iterate in `dump_order` when dumping so observers
        // see a deterministic emission order, and in original order
        // otherwise.
        let patch_iter: &[usize] = if dump { &dump_order } else { &indices[..] };
        for &idx in patch_iter {
            let delayed = all_delayed[idx];
            let instance = delayed.instance;
            // Skip Instances already known to be non-canonical.
            if canon_map.contains_key(&instance) {
                continue;
            }

            let base = instance.strip_outlives(tcx);
            let body = tcx.instance_mir(base.def);
            let mut patched = body.clone();

            if dump {
                let instance_name = with_no_trimmed_paths!(instance.to_string());
                eprintln!("      {instance_name}");
                if delayed.callee_substitutions.is_empty() {
                    eprintln!("        unchanged");
                }
            }

            // Apply callee substitutions, resolving through canon_map.
            for &(call_id, callee) in delayed.callee_substitutions {
                let canonical_callee = canon_map.get(&callee).copied().unwrap_or(callee);
                crate::collector::patch_call_terminator(
                    &mut patched,
                    call_id,
                    canonical_callee,
                    tcx,
                );
                if dump {
                    let summary = crate::cast_sensitivity::format_call_id_summary(tcx, call_id);
                    let canonical_name = with_no_trimmed_paths!(canonical_callee.to_string());
                    eprintln!("        substitution: {summary} -> {canonical_name}");
                }
            }

            // Resolve intrinsic calls in-place.
            crate::resolved_bodies::patch_intrinsic_calls(&mut patched, tcx, instance, resolutions);

            let patched = tcx.arena.alloc(patched);
            patched_bodies.insert(instance, patched);
        }

        // --- Phase 2: Transitive deduplication at this depth ---
        //
        // Group patched Instances by (def_id, base_args). Within each
        // group, check if all callee_substitutions resolve to the
        // same canonical Instances. If so, pick one canonical
        // (smallest OutlivesClass), map the rest → canonical.
        let mut by_base: FxHashMap<(DefId, ty::GenericArgsRef<'tcx>), Vec<usize>> =
            FxHashMap::default();
        for &idx in indices {
            let instance = all_delayed[idx].instance;
            if canon_map.contains_key(&instance) {
                continue;
            }
            let base = instance.strip_outlives(tcx);
            by_base.entry((base.def_id(), base.args)).or_default().push(idx);
        }

        if dump {
            eprintln!("    Phase 2 (dedup):");
        }

        #[allow(rustc::potential_query_instability)]
        for (_key, group) in &by_base {
            if group.len() <= 1 {
                continue;
            }

            // Two Instances in this group are equivalent if their
            // resolved callee sets (after canon_map lookup) are
            // identical. Build a signature for each: the sorted
            // list of (call_id, canonical_callee) pairs.
            //
            // DefId is !Ord and Instance is !Ord, so we sort via
            // StableHasher fingerprints (the idiomatic rustc
            // approach — see ToStableHashKey impls for DefId and
            // Instance).
            let mut sig_groups: FxIndexMap<
                Vec<(&'tcx ty::List<(DefId, u32, ty::GenericArgsRef<'tcx>)>, ty::Instance<'tcx>)>,
                Vec<usize>,
            > = FxIndexMap::default();
            for &idx in group {
                let delayed = all_delayed[idx];
                let mut sig: Vec<_> = delayed
                    .callee_substitutions
                    .iter()
                    .map(|&(call_id, callee)| {
                        let canonical_callee = canon_map.get(&callee).copied().unwrap_or(callee);
                        (call_id, canonical_callee)
                    })
                    .collect();
                tcx.with_stable_hashing_context(|mut hcx| {
                    sig.sort_by_cached_key(|&(call_id, canonical)| {
                        use rustc_data_structures::fingerprint::Fingerprint;
                        use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
                        let mut hasher = StableHasher::new();
                        call_id.hash_stable(&mut hcx, &mut hasher);
                        canonical.hash_stable(&mut hcx, &mut hasher);
                        hasher.finish::<Fingerprint>()
                    });
                });
                sig_groups.entry(sig).or_default().push(idx);
            }

            for (_sig, equiv) in &sig_groups {
                if equiv.len() <= 1 {
                    continue;
                }
                // Pick canonical: smallest OutlivesClass under StableOrd.
                let canonical_idx = *equiv
                    .iter()
                    .min_by_key(|&&idx| OutlivesClass::from_instance(all_delayed[idx].instance))
                    .unwrap();
                let canonical_instance = all_delayed[canonical_idx].instance;
                if dump {
                    let canonical_name = with_no_trimmed_paths!(canonical_instance.to_string());
                    eprintln!("      signature group (size={}):", equiv.len());
                    eprintln!("        canonical: {canonical_name}");
                    // Sort redirected entries deterministically.
                    let mut redirected: Vec<ty::Instance<'tcx>> = equiv
                        .iter()
                        .filter(|&&idx| idx != canonical_idx)
                        .map(|&idx| all_delayed[idx].instance)
                        .collect();
                    tcx.with_stable_hashing_context(|mut hcx| {
                        redirected.sort_by_cached_key(|inst| {
                            use rustc_data_structures::fingerprint::Fingerprint;
                            use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
                            let mut hasher = StableHasher::new();
                            inst.hash_stable(&mut hcx, &mut hasher);
                            hasher.finish::<Fingerprint>()
                        });
                    });
                    for inst in &redirected {
                        let name = with_no_trimmed_paths!(inst.to_string());
                        eprintln!("        redirected: {name}");
                    }
                }
                for &idx in equiv {
                    if idx == canonical_idx {
                        continue;
                    }
                    let instance = all_delayed[idx].instance;
                    let prev = canon_map.insert(instance, canonical_instance);
                    debug_assert!(
                        prev.is_none(),
                        "Instance in multiple signature groups: {instance:?}"
                    );
                    patched_bodies.remove(&instance);
                }
            }
        }

        // --- Phase 3: Feed bodies and insert MonoItems ---
        let mut fed = 0usize;
        let mut skipped = 0usize;
        let mut inserted = 0usize;
        for &idx in indices {
            let delayed = all_delayed[idx];
            let instance = delayed.instance;

            if canon_map.contains_key(&instance) {
                // Non-canonical: callers have been rewritten during the
                // patch pass to call the canonical Instance directly.
                // This Instance is unreachable — skip it.
                skipped += 1;
                continue;
            } else if let Some(body) = patched_bodies.get(&instance) {
                // Canonical (or non-condensed): feed the resolved body.
                tcx.feed_codegen_mir(instance, body);
                fed += 1;
            }
            new_mono_items.push(MonoItem::Fn(instance));
            inserted += 1;
        }

        if dump {
            eprintln!(
                "    Phase 3 (emit):\n      fed: {fed}, skipped (non-canonical): {skipped}, \
                 newly-inserted mono items: {inserted}"
            );
        }
    }

    if dump {
        eprintln!("  Canon map summary:");
        eprintln!("    total redirections: {}", canon_map.len());
    }

    // Insert the resolved MonoItem::Fn entries into mono_items.
    if !new_mono_items.is_empty() {
        let items = mono_items.to_mut();
        items.extend(new_mono_items);
    }
}

/// Seed `canon_map` from leaf-level condensation groups. Groups
/// directly-sensitive Instances by `(def_id, base_args)`, then checks
/// whether their `OutlivesClass`es share the same condensation slot
/// in `trait_cast_layout`. Instances that share a slot are equivalent;
/// the smallest `OutlivesClass` under `StableOrd` is canonical.
fn seed_canon_map_from_condensation<'tcx>(
    tcx: TyCtxt<'tcx>,
    all_delayed: &[&rustc_middle::mono::DelayedInstance<'tcx>],
    canon_map: &mut FxHashMap<ty::Instance<'tcx>, ty::Instance<'tcx>>,
) {
    use rustc_data_structures::stable_hasher::StableCompare;
    use rustc_middle::ty::trait_cast::{FingerprintedTy, IntrinsicSiteKind, OutlivesClass};
    use smallvec::SmallVec;

    // Group leaf Instances (empty callee_substitutions) by (def_id, base_args).
    let mut by_base: FxHashMap<
        (DefId, ty::GenericArgsRef<'tcx>),
        Vec<&rustc_middle::mono::DelayedInstance<'tcx>>,
    > = FxHashMap::default();
    for delayed in all_delayed {
        if !delayed.callee_substitutions.is_empty() {
            continue; // Only seed from leaves.
        }
        let base = delayed.instance.strip_outlives(tcx);
        by_base.entry((base.def_id(), base.args)).or_default().push(delayed);
    }

    #[allow(rustc::potential_query_instability)]
    for (_key, group) in &by_base {
        if group.len() <= 1 {
            continue;
        }

        // Signature: sorted list of (super_trait, sub_trait, slot) for all
        // Index intrinsics in this caller. Two callers with identical
        // signatures produce identical patched bodies.
        //
        // Uses FingerprintedTy for deterministic sorting (Ty is !Ord).
        // Eq/Hash on FingerprintedTy delegate to Ty (interned pointer),
        // so HashMap grouping works identically.
        let mut by_signature: FxHashMap<
            SmallVec<[(FingerprintedTy<'tcx>, FingerprintedTy<'tcx>, usize); 1]>,
            Vec<ty::Instance<'tcx>>,
        > = FxHashMap::default();

        for delayed in group {
            let instance = delayed.instance;
            let mut sig: SmallVec<[(FingerprintedTy<'tcx>, FingerprintedTy<'tcx>, usize); 1]> =
                SmallVec::new();
            for &intrinsic_instance in delayed.intrinsic_callees {
                let site =
                    crate::trait_cast_requests::classify_intrinsic_site(tcx, intrinsic_instance);
                if let IntrinsicSiteKind::Index { super_trait, sub_trait } = site {
                    let outlives_class = trait_metadata_index_outlives_class(
                        tcx,
                        super_trait,
                        sub_trait,
                        intrinsic_instance,
                    );
                    let layout = tcx.trait_cast_layout(super_trait);
                    if let Some(&slot) = layout.index_map.get(&(sub_trait, outlives_class)) {
                        sig.push((
                            FingerprintedTy::new(tcx, super_trait),
                            FingerprintedTy::new(tcx, sub_trait),
                            slot,
                        ));
                    }
                }
            }
            sig.sort_by(|a, b| {
                a.2.cmp(&b.2).then_with(|| a.0.stable_cmp(&b.0)).then_with(|| a.1.stable_cmp(&b.1))
            });
            by_signature.entry(sig).or_default().push(instance);
        }

        // Within each slot group, pick canonical and map the rest.
        //
        // Iteration order of `by_signature` (FxHashMap) is non-deterministic,
        // but this is safe: signature groups *partition* the Instance space —
        // each Instance appears in exactly one bucket (one signature per
        // delayed Instance, one push per delayed Instance). Because groups
        // are disjoint, `canon_map.insert` never overwrites an entry written
        // by a different group, so the final map contents are identical
        // regardless of iteration order. Within each group, `min_by_key`
        // over `OutlivesClass` is deterministic because the `instances` Vec
        // inherits insertion order from `all_delayed` (a slice), and
        // `min_by_key` returns the first minimum on ties.
        #[allow(rustc::potential_query_instability)]
        for (_, instances) in &by_signature {
            if instances.len() <= 1 {
                continue;
            }
            let canonical =
                *instances.iter().min_by_key(|inst| OutlivesClass::from_instance(**inst)).unwrap();
            for &inst in instances {
                if inst == canonical {
                    continue;
                }
                let prev = canon_map.insert(inst, canonical);
                debug_assert!(prev.is_none(), "Instance in multiple signature groups: {inst:?}");
            }
        }
    }
}

/// Query provider: collects mono items for the local crate, including
/// sensitivity analysis and augmentation, but does NOT perform global
/// trait-cast resolution or partitioning.
fn collect_local_mono_items(tcx: TyCtxt<'_>, (): ()) -> LocalMonoItemCollection<'_> {
    let collection_strategy = if tcx.sess.link_dead_code() {
        MonoItemCollectionStrategy::Eager
    } else {
        MonoItemCollectionStrategy::Lazy
    };

    let collection_result = collector::collect_crate_mono_items(tcx, collection_strategy);
    let items = collection_result.mono_items;
    let usage_map = collection_result.usage_map;

    // Perform checks that need to operate on the entire mono item graph.
    target_specific_checks(tcx, &items, &usage_map);

    // If there was an error during collection (e.g. from one of the constants we evaluated),
    // then we stop here. This way codegen does not have to worry about failing constants.
    // (codegen relies on this and ICEs will happen if this is violated.)
    tcx.dcx().abort_if_errors();

    LocalMonoItemCollection {
        mono_items: tcx.arena.alloc_from_iter(items),
        usage_map: tcx.arena.alloc(usage_map),
        delayed_codegen: collection_result.delayed_codegen,
        sensitivity_map: collection_result.sensitivity_map,
    }
}

fn collect_and_partition_mono_items(tcx: TyCtxt<'_>, (): ()) -> MonoItemPartitions<'_> {
    let collection = tcx.collect_local_mono_items(());
    let mut items: Cow<'_, [MonoItem<'_>]> = Cow::Borrowed(collection.mono_items);
    let usage_map = collection.usage_map;

    // Global phase: resolve trait-cast delayed codegen requests into
    // MonoItem::Fn entries before partitioning distributes items.
    tcx.sess.time("resolve_trait_cast_globals", || resolve_trait_cast_globals(tcx, &mut items));

    let (codegen_units, _) = tcx.sess.time("partition_and_assert_distinct_symbols", || {
        par_join(
            || {
                let mut codegen_units = partition(tcx, items.iter().copied(), usage_map);
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
    if let SwitchWithOptPath::Enabled(ref path) = tcx.sess.opts.unstable_opts.dump_mono_stats
        && let Err(err) =
            dump_mono_items_stats(tcx, codegen_units, path, tcx.crate_name(LOCAL_CRATE))
    {
        tcx.dcx().emit_fatal(CouldntDumpMonoStats { error: err.to_string() });
    }

    dump_trait_graph(tcx);
    print_trait_cast_stats(tcx);

    if tcx.sess.opts.unstable_opts.print_mono_items {
        let mut item_to_cgus: UnordMap<_, Vec<_>> = Default::default();

        for cgu in codegen_units {
            for (&mono_item, &data) in cgu.items() {
                item_to_cgus.entry(mono_item).or_default().push((cgu.name(), data.linkage));
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
                for &(ref cgu_name, linkage) in cgus.iter() {
                    output.push(' ');
                    output.push_str(cgu_name.as_str());

                    let linkage_abbrev = match linkage {
                        Linkage::External => "External",
                        Linkage::AvailableExternally => "Available",
                        Linkage::LinkOnceAny => "OnceAny",
                        Linkage::LinkOnceODR => "OnceODR",
                        Linkage::WeakAny => "WeakAny",
                        Linkage::WeakODR => "WeakODR",
                        Linkage::Internal => "Internal",
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

    MonoItemPartitions {
        all_mono_items: tcx.arena.alloc(mono_items),
        codegen_units,
        delayed_codegen: collection.delayed_codegen,
        sensitivity_map: collection.sensitivity_map,
    }
}

/// Outputs stats about instantiation counts and estimated size, per `MonoItem`'s
/// def, to a file in the given output directory.
fn dump_mono_items_stats<'tcx>(
    tcx: TyCtxt<'tcx>,
    codegen_units: &[CodegenUnit<'tcx>],
    output_directory: &Option<PathBuf>,
    crate_name: Symbol,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_directory = if let Some(directory) = output_directory {
        fs::create_dir_all(directory)?;
        directory
    } else {
        Path::new(".")
    };

    let format = tcx.sess.opts.unstable_opts.dump_mono_stats_format;
    let ext = format.extension();
    let filename = format!("{crate_name}.mono_items.{ext}");
    let output_path = output_directory.join(&filename);
    let mut file = File::create_buffered(&output_path)?;

    // Gather instantiated mono items grouped by def_id
    let mut items_per_def_id: FxIndexMap<_, Vec<_>> = Default::default();
    for cgu in codegen_units {
        cgu.items()
            .keys()
            // Avoid variable-sized compiler-generated shims
            .filter(|mono_item| mono_item.is_user_defined())
            .for_each(|mono_item| {
                items_per_def_id.entry(mono_item.def_id()).or_default().push(mono_item);
            });
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

/// Dump trait graph info for root supertraits matching `-Z dump-trait-graph`.
fn dump_trait_graph(tcx: TyCtxt<'_>) {
    let Some(ref filter) = tcx.sess.opts.unstable_opts.dump_trait_graph else {
        return;
    };

    let requests = tcx.gather_trait_cast_requests(());
    if requests.is_empty() {
        return;
    }

    use rustc_middle::ty::trait_cast::FingerprintedTy;

    let roots: Vec<_> = requests
        .root_traits()
        .into_items()
        .map(|ty| FingerprintedTy::new(tcx, ty))
        .into_sorted_stable_ord();

    for fp_root in &roots {
        let root = fp_root.ty();
        let root_str = with_no_trimmed_paths!(root.to_string());
        if filter != "all" && !root_str.contains(filter.as_str()) {
            continue;
        }

        let graph = tcx.trait_cast_graph(root);
        let layout = tcx.trait_cast_layout(root);

        eprintln!("=== Trait Graph: {root_str} ===");

        // Sub-traits + outlives classes.
        let sub_traits = graph
            .sub_traits
            .items()
            .map(|(k, v)| (*k, v))
            .into_sorted_stable_ord_by_key(|item| &item.0);
        eprintln!("  Sub-traits ({}):", sub_traits.len());
        for (fp_ty, info) in &sub_traits {
            let classes: Vec<_> = info.outlives_classes.items().copied().into_sorted_stable_ord();
            let sub_str = with_no_trimmed_paths!(fp_ty.ty().to_string());
            eprintln!("    {sub_str} — {} outlives class(es)", classes.len());
            for (i, cls) in classes.iter().enumerate() {
                let pairs: Vec<String> = cls.iter().map(|(l, s)| format!("('{l}: '{s})")).collect();
                let pairs_str =
                    if pairs.is_empty() { "empty".to_string() } else { pairs.join(", ") };
                eprintln!("      [{i}] {{{pairs_str}}}");
            }
        }

        // Concrete types.
        let concretes: Vec<_> = graph.concrete_types.items().copied().into_sorted_stable_ord();
        eprintln!("  Concrete types ({}):", concretes.len());
        for ct in &concretes {
            let ct_str = with_no_trimmed_paths!(ct.ty().to_string());
            eprintln!("    {ct_str}");
        }

        // Table layout.
        eprintln!("  Table layout: {} slot(s)", layout.table_length);
        for (idx, si) in layout.slot_info.iter().enumerate() {
            let pairs: Vec<String> =
                si.outlives_class.iter().map(|(l, s)| format!("('{l}: '{s})")).collect();
            let pairs_str = if pairs.is_empty() { "empty".to_string() } else { pairs.join(", ") };
            let sub_str = with_no_trimmed_paths!(si.sub_trait.to_string());
            eprintln!("    slot[{idx}]: sub={sub_str}, bvs={}, class={{{pairs_str}}}", si.num_bvs);
        }

        // Condensation summary (only when classes were collapsed).
        for (fp_ty, info) in &sub_traits {
            let raw_classes = info.outlives_classes.len();
            let slots = layout.slot_info.iter().filter(|si| si.sub_trait == **fp_ty).count();
            if raw_classes != slots {
                let sub_str = with_no_trimmed_paths!(fp_ty.ty().to_string());
                eprintln!("  Condensation: {sub_str} — {raw_classes} class(es) -> {slots} slot(s)");
            }
        }

        // Admissibility per (concrete_type, sub_trait).
        {
            use crate::trait_graph::resolve_dyn_satisfaction;
            let mut any = false;
            for ct in &concretes {
                for (fp_ty, _) in &sub_traits {
                    if let Some(impl_def_id) = resolve_dyn_satisfaction(tcx, **ct, **fp_ty) {
                        if !any {
                            eprintln!("  Admissibility:");
                            any = true;
                        }
                        let ua = tcx.impl_universally_admissible(impl_def_id);
                        let ct_str = with_no_trimmed_paths!(ct.ty().to_string());
                        let sub_str = with_no_trimmed_paths!(fp_ty.ty().to_string());
                        let impl_str = tcx.def_path_str(impl_def_id);
                        eprintln!(
                            "    {ct_str} : {sub_str} — impl {impl_str} \
                             (univ_admissible={ua})"
                        );
                    }
                }
            }
        }

        eprintln!();
    }
}

/// Print summary statistics for the trait-cast monomorphization pipeline to
/// stderr, gated on `-Z print-trait-cast-stats`. Emits a single compact block
/// derived from query results already computed by the partitioning pass, so
/// this is effectively free when the flag is off.
fn print_trait_cast_stats(tcx: TyCtxt<'_>) {
    if !tcx.sess.opts.unstable_opts.print_trait_cast_stats {
        return;
    }

    // Gather all delayed codegen entries across crates, deduplicating by
    // `Instance` (mirrors the dedup pattern in `cascade_canonicalize`).
    let mut seen: FxHashSet<ty::Instance<'_>> = FxHashSet::default();
    let mut delayed_total = 0usize;
    let mut augmented = 0usize;
    let mut intrinsic_sites = 0usize;
    for delayed in tcx.delayed_codegen_requests(LOCAL_CRATE) {
        if seen.insert(delayed.instance) {
            delayed_total += 1;
            if delayed.instance.has_outlives_entries() {
                augmented += 1;
            }
            intrinsic_sites += delayed.intrinsic_callees.len();
        }
    }
    for &cnum in tcx.crates(()) {
        for delayed in tcx.delayed_codegen_requests(cnum) {
            if seen.insert(delayed.instance) {
                delayed_total += 1;
                if delayed.instance.has_outlives_entries() {
                    augmented += 1;
                }
                intrinsic_sites += delayed.intrinsic_callees.len();
            }
        }
    }

    let requests = tcx.gather_trait_cast_requests(());
    let roots = requests.root_traits();
    let root_count = roots.len();

    // Iteration order over an `UnordSet` doesn't matter for a sum.
    let total_slots: usize =
        roots.items().map(|root| tcx.trait_cast_layout(*root).table_length).sum();

    eprintln!("trait-cast stats:");
    eprintln!("  delayed codegen entries:        {delayed_total}");
    eprintln!(
        "  augmented instances:            {augmented}   \
         (instances with outlives entries among delayed)"
    );
    eprintln!(
        "  trait-cast intrinsic sites:     {intrinsic_sites}   \
         (sum over delayed instances)"
    );
    eprintln!(
        "  root supertraits:               {root_count}   \
         (from gather_trait_cast_requests.root_traits())"
    );
    eprintln!(
        "  total table slots:              {total_slots}   \
         (sum of trait_cast_layout(root).table_length over roots)"
    );
}

/// Query provider for `is_transitively_delayed_instance`.
///
/// Compares on the strip-outlives form. The mono collector's
/// `augment_sensitive_subgraphs` pushes *augmented* instances (those carrying
/// the `OUTLIVES_SENTINEL` or real outlives entries) into `delayed_codegen`,
/// while MIR call sites — including the ones codegen re-mangles on a cache
/// miss in `get_fn` — may reach this query with the pre-augmentation base
/// Instance. The v0 mangler's impl-path does not include Outlives args in
/// the emitted symbol, so augmented and base share a mangled name when the
/// instantiating-crate suffix is suppressed; for the suffix-stripping
/// mangler gate to apply uniformly, both forms must report as delayed.
///
/// Membership is a single O(1) lookup against
/// `delayed_codegen_stripped_set(())`, which flattens every crate's
/// delayed-codegen set into one precomputed `UnordSet`.
fn is_transitively_delayed_instance_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
) -> bool {
    // Metadata-only builds (rustdoc, `--emit=metadata`) don't run mono
    // collection, so `delayed_codegen_requests(LOCAL_CRATE)` — which
    // forces `collect_local_mono_items` — is both meaningless and
    // impossible to satisfy (collection demands upstream `optimized_mir`
    // that the metadata-only pipeline won't have loaded). Return
    // `false` conservatively: no local crate can register delayed
    // instances without a codegen phase, and the mangler's suffix-
    // stripping gate is a no-op for those pathways anyway (the emitted
    // metadata records DefId+args, not the pre-mangled name).
    if !tcx.sess.opts.output_types.should_codegen() {
        return false;
    }
    let stripped = instance.strip_outlives(tcx);
    tcx.delayed_codegen_stripped_set(()).contains(&stripped)
}

fn delayed_codegen_stripped_set_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> UnordSet<ty::Instance<'tcx>> {
    use rustc_hir::def_id::LOCAL_CRATE;
    // An instance's `def_id().krate` is the crate that *defines* the
    // generic, not where it's monomorphized — e.g. the blanket-impl
    // `<T as TraitMetadataTable<I>>::derived_metadata_table` mono for
    // `T = cross_crate_lib::LibTypeA` has `def_id().krate == core` yet
    // is classified delayed when upstream `cross_crate_lib` collects
    // it. Flatten every crate's set so callers don't have to scan.
    let mut set = UnordSet::default();
    for d in tcx.delayed_codegen_requests(LOCAL_CRATE) {
        set.insert(d.instance.strip_outlives(tcx));
    }
    for &cnum in tcx.crates(()) {
        for d in tcx.delayed_codegen_requests(cnum) {
            set.insert(d.instance.strip_outlives(tcx));
        }
    }
    set
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.queries.collect_local_mono_items = collect_local_mono_items;
    providers.queries.collect_and_partition_mono_items = collect_and_partition_mono_items;

    // These project from collect_local_mono_items (NOT collect_and_partition_mono_items)
    // to avoid a query cycle: collect_and_partition_mono_items → gather_trait_cast_requests
    // → delayed_codegen_requests → collect_and_partition_mono_items.
    providers.queries.delayed_codegen_requests = |tcx, _key: rustc_middle::query::LocalCrate| {
        tcx.collect_local_mono_items(()).delayed_codegen
    };

    providers.queries.crate_cast_relevant_lifetimes =
        |tcx, _key: rustc_middle::query::LocalCrate| {
            let collection = tcx.collect_local_mono_items(());
            collection.sensitivity_map
        };

    providers.queries.cast_relevant_lifetimes = |tcx, instance| {
        let map = tcx.crate_cast_relevant_lifetimes(instance.def_id().krate);
        map.get(&instance)
    };

    // Local provider: project the LocalDefId set from delayed_codegen.
    // Consumed by the rmeta encoder's `should_encode_mir` gate so that
    // transitively-delayed non-generic fns (e.g. user fns whose only
    // intrinsic reach is via post-monomorphization inlining of the
    // `core::TraitCast` trampolines) ship their MIR downstream.
    providers.queries.local_def_ids_backing_delayed_instances = |tcx, _: ()| {
        let delayed = tcx.collect_local_mono_items(()).delayed_codegen;
        let mut set = rustc_hir::def_id::LocalDefIdSet::default();
        for d in delayed.iter() {
            if let Some(local_def_id) = d.instance.def_id().as_local() {
                set.insert(local_def_id);
            }
        }
        tcx.arena.alloc(set)
    };

    providers.queries.is_transitively_delayed_instance = is_transitively_delayed_instance_provider;
    providers.queries.delayed_codegen_stripped_set = delayed_codegen_stripped_set_provider;

    providers.queries.is_codegened_item =
        |tcx, def_id| tcx.collect_and_partition_mono_items(()).all_mono_items.contains(&def_id);

    providers.queries.codegen_unit = |tcx, name| {
        tcx.collect_and_partition_mono_items(())
            .codegen_units
            .iter()
            .find(|cgu| cgu.name() == name)
            .unwrap_or_else(|| panic!("failed to find cgu with name {name:?}"))
    };

    providers.queries.size_estimate = |tcx, instance| {
        match instance.def {
            // "Normal" functions size estimate: the number of
            // statements, plus one for the terminator.
            InstanceKind::Item(..)
            | InstanceKind::DropGlue(..)
            | InstanceKind::AsyncDropGlueCtorShim(..) => {
                let mir = tcx.instance_mir(instance.def);
                mir.basic_blocks
                    .iter()
                    .map(|bb| {
                        bb.statements
                            .iter()
                            .filter_map(|stmt| match stmt.kind {
                                StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                                    None
                                }
                                _ => Some(stmt),
                            })
                            .count()
                            + 1
                    })
                    .sum()
            }
            // Other compiler-generated shims size estimate: 1
            _ => 1,
        }
    };

    collector::provide(providers);
}
