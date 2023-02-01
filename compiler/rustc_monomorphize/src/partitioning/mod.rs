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

mod default;
mod merging;

use std::cmp;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync;
use rustc_hir::def_id::{DefIdSet, LOCAL_CRATE};
use rustc_middle::mir;
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::mir::mono::{CodegenUnit, Linkage};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{DumpMonoStatsFormat, SwitchWithOptPath};
use rustc_span::symbol::Symbol;

use crate::collector::InliningMap;
use crate::collector::{self, MonoItemCollectionMode};
use crate::errors::{
    CouldntDumpMonoStats, SymbolAlreadyDefined, UnknownCguCollectionMode, UnknownPartitionStrategy,
};

pub struct PartitioningCx<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    target_cgu_count: usize,
    inlining_map: &'a InliningMap<'tcx>,
}

trait Partitioner<'tcx> {
    fn place_root_mono_items(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        mono_items: &mut dyn Iterator<Item = MonoItem<'tcx>>,
    ) -> PreInliningPartitioning<'tcx>;

    fn merge_codegen_units(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        initial_partitioning: &mut PreInliningPartitioning<'tcx>,
    );

    fn place_inlined_mono_items(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        initial_partitioning: PreInliningPartitioning<'tcx>,
    ) -> PostInliningPartitioning<'tcx>;

    fn internalize_symbols(
        &mut self,
        cx: &PartitioningCx<'_, 'tcx>,
        partitioning: &mut PostInliningPartitioning<'tcx>,
    );
}

fn get_partitioner<'tcx>(tcx: TyCtxt<'tcx>) -> Box<dyn Partitioner<'tcx>> {
    let strategy = match &tcx.sess.opts.unstable_opts.cgu_partitioning_strategy {
        None => "default",
        Some(s) => &s[..],
    };

    match strategy {
        "default" => Box::new(default::DefaultPartitioning),
        _ => {
            tcx.sess.emit_fatal(UnknownPartitionStrategy);
        }
    }
}

pub fn partition<'tcx>(
    tcx: TyCtxt<'tcx>,
    mono_items: &mut dyn Iterator<Item = MonoItem<'tcx>>,
    max_cgu_count: usize,
    inlining_map: &InliningMap<'tcx>,
) -> Vec<CodegenUnit<'tcx>> {
    let _prof_timer = tcx.prof.generic_activity("cgu_partitioning");

    let mut partitioner = get_partitioner(tcx);
    let cx = &PartitioningCx { tcx, target_cgu_count: max_cgu_count, inlining_map };
    // In the first step, we place all regular monomorphizations into their
    // respective 'home' codegen unit. Regular monomorphizations are all
    // functions and statics defined in the local crate.
    let mut initial_partitioning = {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_place_roots");
        partitioner.place_root_mono_items(cx, mono_items)
    };

    initial_partitioning.codegen_units.iter_mut().for_each(|cgu| cgu.create_size_estimate(tcx));

    debug_dump(tcx, "INITIAL PARTITIONING:", initial_partitioning.codegen_units.iter());

    // Merge until we have at most `max_cgu_count` codegen units.
    {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_merge_cgus");
        partitioner.merge_codegen_units(cx, &mut initial_partitioning);
        debug_dump(tcx, "POST MERGING:", initial_partitioning.codegen_units.iter());
    }

    // In the next step, we use the inlining map to determine which additional
    // monomorphizations have to go into each codegen unit. These additional
    // monomorphizations can be drop-glue, functions from external crates, and
    // local functions the definition of which is marked with `#[inline]`.
    let mut post_inlining = {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_place_inline_items");
        partitioner.place_inlined_mono_items(cx, initial_partitioning)
    };

    post_inlining.codegen_units.iter_mut().for_each(|cgu| cgu.create_size_estimate(tcx));

    debug_dump(tcx, "POST INLINING:", post_inlining.codegen_units.iter());

    // Next we try to make as many symbols "internal" as possible, so LLVM has
    // more freedom to optimize.
    if !tcx.sess.link_dead_code() {
        let _prof_timer = tcx.prof.generic_activity("cgu_partitioning_internalize_symbols");
        partitioner.internalize_symbols(cx, &mut post_inlining);
    }

    let instrument_dead_code =
        tcx.sess.instrument_coverage() && !tcx.sess.instrument_coverage_except_unused_functions();

    if instrument_dead_code {
        assert!(
            post_inlining.codegen_units.len() > 0,
            "There must be at least one CGU that code coverage data can be generated in."
        );

        // Find the smallest CGU that has exported symbols and put the dead
        // function stubs in that CGU. We look for exported symbols to increase
        // the likelihood the linker won't throw away the dead functions.
        // FIXME(#92165): In order to truly resolve this, we need to make sure
        // the object file (CGU) containing the dead function stubs is included
        // in the final binary. This will probably require forcing these
        // function symbols to be included via `-u` or `/include` linker args.
        let mut cgus: Vec<_> = post_inlining.codegen_units.iter_mut().collect();
        cgus.sort_by_key(|cgu| cgu.size_estimate());

        let dead_code_cgu =
            if let Some(cgu) = cgus.into_iter().rev().find(|cgu| {
                cgu.items().iter().any(|(_, (linkage, _))| *linkage == Linkage::External)
            }) {
                cgu
            } else {
                // If there are no CGUs that have externally linked items,
                // then we just pick the first CGU as a fallback.
                &mut post_inlining.codegen_units[0]
            };
        dead_code_cgu.make_code_coverage_dead_code_cgu();
    }

    // Finally, sort by codegen unit name, so that we get deterministic results.
    let PostInliningPartitioning {
        codegen_units: mut result,
        mono_item_placements: _,
        internalization_candidates: _,
    } = post_inlining;

    result.sort_by(|a, b| a.name().as_str().partial_cmp(b.name().as_str()).unwrap());

    result
}

pub struct PreInliningPartitioning<'tcx> {
    codegen_units: Vec<CodegenUnit<'tcx>>,
    roots: FxHashSet<MonoItem<'tcx>>,
    internalization_candidates: FxHashSet<MonoItem<'tcx>>,
}

/// For symbol internalization, we need to know whether a symbol/mono-item is
/// accessed from outside the codegen unit it is defined in. This type is used
/// to keep track of that.
#[derive(Clone, PartialEq, Eq, Debug)]
enum MonoItemPlacement {
    SingleCgu { cgu_name: Symbol },
    MultipleCgus,
}

struct PostInliningPartitioning<'tcx> {
    codegen_units: Vec<CodegenUnit<'tcx>>,
    mono_item_placements: FxHashMap<MonoItem<'tcx>, MonoItemPlacement>,
    internalization_candidates: FxHashSet<MonoItem<'tcx>>,
}

fn debug_dump<'a, 'tcx, I>(tcx: TyCtxt<'tcx>, label: &str, cgus: I)
where
    I: Iterator<Item = &'a CodegenUnit<'tcx>>,
    'tcx: 'a,
{
    let dump = move || {
        use std::fmt::Write;

        let s = &mut String::new();
        let _ = writeln!(s, "{label}");
        for cgu in cgus {
            let _ =
                writeln!(s, "CodegenUnit {} estimated size {} :", cgu.name(), cgu.size_estimate());

            for (mono_item, linkage) in cgu.items() {
                let symbol_name = mono_item.symbol_name(tcx).name;
                let symbol_hash_start = symbol_name.rfind('h');
                let symbol_hash = symbol_hash_start.map_or("<no hash>", |i| &symbol_name[i..]);

                let _ = writeln!(
                    s,
                    " - {} [{:?}] [{}] estimated size {}",
                    mono_item,
                    linkage,
                    symbol_hash,
                    mono_item.size_estimate(tcx)
                );
            }

            let _ = writeln!(s);
        }

        std::mem::take(s)
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

    let (items, inlining_map) = collector::collect_crate_mono_items(tcx, collection_mode);

    tcx.sess.abort_if_errors();

    let (codegen_units, _) = tcx.sess.time("partition_and_assert_distinct_symbols", || {
        sync::join(
            || {
                let mut codegen_units = partition(
                    tcx,
                    &mut items.iter().cloned(),
                    tcx.sess.codegen_units(),
                    &inlining_map,
                );
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

/// Outputs stats about instantation counts and estimated size, per `MonoItem`'s
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
