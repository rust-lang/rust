use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, ConstCodegenMethods};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::mir;
use rustc_middle::mir::mono::MonoItemPartitions;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefIdSet;

use crate::common::CodegenCx;
use crate::coverageinfo::mapgen::covfun::{CovfunRecord, prepare_covfun_record};
use crate::llvm;

/// Each CGU will normally only emit coverage metadata for the functions that it actually generates.
/// But since we don't want unused functions to disappear from coverage reports, we also scan for
/// functions that were instrumented but are not participating in codegen.
///
/// These unused functions don't need to be codegenned, but we do need to add them to the function
/// coverage map (in a single designated CGU) so that we still emit coverage mappings for them.
/// We also end up adding their symbol names to a special global array that LLVM will include in
/// its embedded coverage data.
pub(crate) fn prepare_covfun_records_for_unused_functions<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    covfun_records: &mut Vec<CovfunRecord<'tcx>>,
) {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let mut unused_instances = gather_unused_function_instances(cx);
    // Sort the unused instances by symbol name, so that their order isn't hash-sensitive.
    unused_instances.sort_by_key(|instance| instance.symbol_name);

    // Try to create a covfun record for each unused function.
    let mut name_globals = Vec::with_capacity(unused_instances.len());
    covfun_records.extend(unused_instances.into_iter().filter_map(|unused| try {
        let record = prepare_covfun_record(cx.tcx, unused.instance, false)?;
        // If successful, also store its symbol name in a global constant.
        name_globals.push(cx.const_str(unused.symbol_name.name).0);
        record
    }));

    // Store the names of unused functions in a specially-named global array.
    // LLVM's `InstrProfilling` pass will detect this array, and include the
    // referenced names in its `__llvm_prf_names` section.
    // (See `llvm/lib/Transforms/Instrumentation/InstrProfiling.cpp`.)
    if !name_globals.is_empty() {
        let initializer = cx.const_array(cx.type_ptr(), &name_globals);

        let array = llvm::add_global(cx.llmod, cx.val_ty(initializer), c"__llvm_coverage_names");
        llvm::set_global_constant(array, true);
        llvm::set_linkage(array, llvm::Linkage::InternalLinkage);
        llvm::set_initializer(array, initializer);
    }
}

/// Holds a dummy function instance along with its symbol name, to avoid having
/// to repeatedly query for the name.
struct UnusedInstance<'tcx> {
    instance: ty::Instance<'tcx>,
    symbol_name: ty::SymbolName<'tcx>,
}

fn gather_unused_function_instances<'tcx>(cx: &CodegenCx<'_, 'tcx>) -> Vec<UnusedInstance<'tcx>> {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let tcx = cx.tcx;
    let usage = prepare_usage_sets(tcx);

    let is_unused_fn = |def_id: LocalDefId| -> bool {
        // Usage sets expect `DefId`, so convert from `LocalDefId`.
        let d: DefId = LocalDefId::to_def_id(def_id);
        // To be potentially eligible for "unused function" mappings, a definition must:
        // - Be eligible for coverage instrumentation
        // - Not participate directly in codegen (or have lost all its coverage statements)
        // - Not have any coverage statements inlined into codegenned functions
        tcx.is_eligible_for_coverage(def_id)
            && (!usage.all_mono_items.contains(&d) || usage.missing_own_coverage.contains(&d))
            && !usage.used_via_inlining.contains(&d)
    };

    // FIXME(#79651): Consider trying to filter out dummy instantiations of
    // unused generic functions from library crates, because they can produce
    // "unused instantiation" in coverage reports even when they are actually
    // used by some downstream crate in the same binary.

    tcx.mir_keys(())
        .iter()
        .copied()
        .filter(|&def_id| is_unused_fn(def_id))
        .map(|def_id| make_dummy_instance(tcx, def_id))
        .map(|instance| UnusedInstance { instance, symbol_name: tcx.symbol_name(instance) })
        .collect::<Vec<_>>()
}

struct UsageSets<'tcx> {
    all_mono_items: &'tcx DefIdSet,
    used_via_inlining: FxHashSet<DefId>,
    missing_own_coverage: FxHashSet<DefId>,
}

/// Prepare sets of definitions that are relevant to deciding whether something
/// is an "unused function" for coverage purposes.
fn prepare_usage_sets<'tcx>(tcx: TyCtxt<'tcx>) -> UsageSets<'tcx> {
    let MonoItemPartitions { all_mono_items, codegen_units, .. } =
        tcx.collect_and_partition_mono_items(());

    // Obtain a MIR body for each function participating in codegen, via an
    // arbitrary instance.
    let mut def_ids_seen = FxHashSet::default();
    let def_and_mir_for_all_mono_fns = codegen_units
        .iter()
        .flat_map(|cgu| cgu.items().keys())
        .filter_map(|item| match item {
            mir::mono::MonoItem::Fn(instance) => Some(instance),
            mir::mono::MonoItem::Static(_) | mir::mono::MonoItem::GlobalAsm(_) => None,
        })
        // We only need one arbitrary instance per definition.
        .filter(move |instance| def_ids_seen.insert(instance.def_id()))
        .map(|instance| {
            // We don't care about the instance, just its underlying MIR.
            let body = tcx.instance_mir(instance.def);
            (instance.def_id(), body)
        });

    // Functions whose coverage statements were found inlined into other functions.
    let mut used_via_inlining = FxHashSet::default();
    // Functions that were instrumented, but had all of their coverage statements
    // removed by later MIR transforms (e.g. UnreachablePropagation).
    let mut missing_own_coverage = FxHashSet::default();

    for (def_id, body) in def_and_mir_for_all_mono_fns {
        let mut saw_own_coverage = false;

        // Inspect every coverage statement in the function's MIR.
        for stmt in body
            .basic_blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter(|stmt| matches!(stmt.kind, mir::StatementKind::Coverage(_)))
        {
            if let Some(inlined) = stmt.source_info.scope.inlined_instance(&body.source_scopes) {
                // This coverage statement was inlined from another function.
                used_via_inlining.insert(inlined.def_id());
            } else {
                // Non-inlined coverage statements belong to the enclosing function.
                saw_own_coverage = true;
            }
        }

        if !saw_own_coverage && body.function_coverage_info.is_some() {
            missing_own_coverage.insert(def_id);
        }
    }

    UsageSets { all_mono_items, used_via_inlining, missing_own_coverage }
}

fn make_dummy_instance<'tcx>(tcx: TyCtxt<'tcx>, local_def_id: LocalDefId) -> ty::Instance<'tcx> {
    let def_id = local_def_id.to_def_id();

    // Make a dummy instance that fills in all generics with placeholders.
    ty::Instance::new(
        def_id,
        ty::GenericArgs::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    )
}
