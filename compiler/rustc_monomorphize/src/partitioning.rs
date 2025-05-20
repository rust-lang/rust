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

mod autodiff;

use std::cmp;
use std::collections::hash_map::Entry;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use rustc_attr_data_structures::InlineAttr;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::sync;
use rustc_data_structures::unord::{UnordMap, UnordSet};
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, DefIdSet, LOCAL_CRATE};
use rustc_hir::definitions::DefPathDataName;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::middle::exported_symbols::{SymbolExportInfo, SymbolExportLevel};
use rustc_middle::mir::mono::{
    CodegenUnit, CodegenUnitNameBuilder, InstantiationMode, Linkage, MonoItem, MonoItemData,
    MonoItemPartitions, Visibility,
};
use rustc_middle::ty::print::{characteristic_def_id_of_type, with_no_trimmed_paths};
use rustc_middle::ty::{self, InstanceKind, TyCtxt};
use rustc_middle::util::Providers;
use rustc_session::CodegenUnits;
use rustc_session::config::{DumpMonoStatsFormat, SwitchWithOptPath};
use rustc_span::Symbol;
use rustc_target::spec::SymbolVisibility;
use tracing::debug;

use crate::collector::{self, MonoItemCollectionStrategy, UsageMap};
use crate::errors::{CouldntDumpMonoStats, SymbolAlreadyDefined};

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
                if !cx.tcx.is_lang_item(mono_item.def_id(), LangItem::Start) {
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

        // We can't differentiate a function that got inlined.
        let autodiff_active = cfg!(llvm_enzyme)
            && matches!(mono_item, MonoItem::Fn(_))
            && cx
                .tcx
                .codegen_fn_attrs(mono_item.def_id())
                .autodiff_item
                .as_ref()
                .is_some_and(|ad| ad.is_active());

        if !autodiff_active && visibility == Visibility::Hidden && can_be_internalized {
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
            // This is a CGU-private copy.
            cgu.items_mut().entry(inlined_item).or_insert_with(|| MonoItemData {
                inlined: true,
                linkage: Linkage::Internal,
                visibility: Visibility::Default,
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

    let mut codegen_units: Vec<_> = cx.tcx.with_stable_hashing_context(|ref hcx| {
        codegen_units.into_items().map(|(_, cgu)| cgu).collect_sorted(hcx, true)
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
                if cx.tcx.sess.opts.unstable_opts.human_readable_cgu_names {
                    cgu.set_name(Symbol::intern(new_cgu_name));
                } else {
                    // If we don't require CGU names to be human-readable,
                    // we use a fixed length hash of the composite CGU name
                    // instead.
                    let new_cgu_name = CodegenUnit::mangle_name(new_cgu_name);
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

            if tcx.trait_of_item(def_id).is_some() {
                let self_ty = instance.args.type_at(0);
                // This is a default implementation of a trait method.
                return characteristic_def_id_of_type(self_ty).or(Some(def_id));
            }

            if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
                if tcx.sess.opts.incremental.is_some()
                    && tcx
                        .trait_id_of_impl(impl_def_id)
                        .is_some_and(|def_id| tcx.is_lang_item(def_id, LangItem::Drop))
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
    //        This may be fixable with a new `InstanceKind` perhaps? Unsure!
    if tcx.is_lang_item(def_id, LangItem::Start) {
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

fn collect_and_partition_mono_items(tcx: TyCtxt<'_>, (): ()) -> MonoItemPartitions<'_> {
    let collection_strategy = if tcx.sess.link_dead_code() {
        MonoItemCollectionStrategy::Eager
    } else {
        MonoItemCollectionStrategy::Lazy
    };

    let (items, usage_map) = collector::collect_crate_mono_items(tcx, collection_strategy);

    // If there was an error during collection (e.g. from one of the constants we evaluated),
    // then we stop here. This way codegen does not have to worry about failing constants.
    // (codegen relies on this and ICEs will happen if this is violated.)
    tcx.dcx().abort_if_errors();

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

    #[cfg(not(llvm_enzyme))]
    let autodiff_mono_items: Vec<_> = vec![];
    #[cfg(llvm_enzyme)]
    let mut autodiff_mono_items: Vec<_> = vec![];
    let mono_items: DefIdSet = items
        .iter()
        .filter_map(|mono_item| match *mono_item {
            MonoItem::Fn(ref instance) => {
                #[cfg(llvm_enzyme)]
                autodiff_mono_items.push((mono_item, instance));
                Some(instance.def_id())
            }
            MonoItem::Static(def_id) => Some(def_id),
            _ => None,
        })
        .collect();

    let autodiff_items =
        autodiff::find_autodiff_source_functions(tcx, &usage_map, autodiff_mono_items);
    let autodiff_items = tcx.arena.alloc_from_iter(autodiff_items);

    // Output monomorphization stats per def_id
    if let SwitchWithOptPath::Enabled(ref path) = tcx.sess.opts.unstable_opts.dump_mono_stats {
        if let Err(err) =
            dump_mono_items_stats(tcx, codegen_units, path, tcx.crate_name(LOCAL_CRATE))
        {
            tcx.dcx().emit_fatal(CouldntDumpMonoStats { error: err.to_string() });
        }
    }

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
        autodiff_items,
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

pub(crate) fn provide(providers: &mut Providers) {
    providers.collect_and_partition_mono_items = collect_and_partition_mono_items;

    providers.is_codegened_item =
        |tcx, def_id| tcx.collect_and_partition_mono_items(()).all_mono_items.contains(&def_id);

    providers.codegen_unit = |tcx, name| {
        tcx.collect_and_partition_mono_items(())
            .codegen_units
            .iter()
            .find(|cgu| cgu.name() == name)
            .unwrap_or_else(|| panic!("failed to find cgu with name {name:?}"))
    };

    providers.size_estimate = |tcx, instance| {
        match instance.def {
            // "Normal" functions size estimate: the number of
            // statements, plus one for the terminator.
            InstanceKind::Item(..)
            | InstanceKind::DropGlue(..)
            | InstanceKind::AsyncDropGlueCtorShim(..) => {
                let mir = tcx.instance_mir(instance.def);
                mir.basic_blocks.iter().map(|bb| bb.statements.len() + 1).sum()
            }
            // Other compiler-generated shims size estimate: 1
            _ => 1,
        }
    };

    collector::provide(providers);
}
