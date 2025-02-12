use std::sync::Arc;

use itertools::Itertools;
use rustc_abi::Align;
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::IndexVec;
use rustc_middle::mir;
use rustc_middle::mir::mono::MonoItemPartitions;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::RemapFileNameExt;
use rustc_session::config::RemapPathScopeComponents;
use rustc_span::def_id::DefIdSet;
use rustc_span::{SourceFile, StableSourceFileId};
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::llvm_cov;
use crate::coverageinfo::mapgen::covfun::prepare_covfun_record;
use crate::llvm;

mod covfun;
mod spans;

/// Generates and exports the coverage map, which is embedded in special
/// linker sections in the final binary.
///
/// Those sections are then read and understood by LLVM's `llvm-cov` tool,
/// which is distributed in the `llvm-tools` rustup component.
pub(crate) fn finalize(cx: &CodegenCx<'_, '_>) {
    let tcx = cx.tcx;

    // Ensure that LLVM is using a version of the coverage mapping format that
    // agrees with our Rust-side code. Expected versions (encoded as n-1) are:
    // - `CovMapVersion::Version7` (6) used by LLVM 18-19
    let covmap_version = {
        let llvm_covmap_version = llvm_cov::mapping_version();
        let expected_versions = 6..=6;
        assert!(
            expected_versions.contains(&llvm_covmap_version),
            "Coverage mapping version exposed by `llvm-wrapper` is out of sync; \
            expected {expected_versions:?} but was {llvm_covmap_version}"
        );
        // This is the version number that we will embed in the covmap section:
        llvm_covmap_version
    };

    debug!("Generating coverage map for CodegenUnit: `{}`", cx.codegen_unit.name());

    // FIXME(#132395): Can this be none even when coverage is enabled?
    let instances_used = match cx.coverage_cx {
        Some(ref cx) => cx.instances_used.borrow(),
        None => return,
    };

    // The order of entries in this global file table needs to be deterministic,
    // and ideally should also be independent of the details of stable-hashing,
    // because coverage tests snapshots (`.cov-map`) can observe the order and
    // would need to be re-blessed if it changes. As long as those requirements
    // are satisfied, the order can be arbitrary.
    let mut global_file_table = GlobalFileTable::new();

    let mut covfun_records = instances_used
        .iter()
        .copied()
        // Sort by symbol name, so that the global file table is built in an
        // order that doesn't depend on the stable-hash-based order in which
        // instances were visited during codegen.
        .sorted_by_cached_key(|&instance| tcx.symbol_name(instance).name)
        .filter_map(|instance| prepare_covfun_record(tcx, &mut global_file_table, instance, true))
        .collect::<Vec<_>>();

    // In a single designated CGU, also prepare covfun records for functions
    // in this crate that were instrumented for coverage, but are unused.
    if cx.codegen_unit.is_code_coverage_dead_code_cgu() {
        let mut unused_instances = gather_unused_function_instances(cx);
        // Sort the unused instances by symbol name, for the same reason as the used ones.
        unused_instances.sort_by_cached_key(|&instance| tcx.symbol_name(instance).name);
        covfun_records.extend(unused_instances.into_iter().filter_map(|instance| {
            prepare_covfun_record(tcx, &mut global_file_table, instance, false)
        }));
    }

    // If there are no covfun records for this CGU, don't generate a covmap record.
    // Emitting a covmap record without any covfun records causes `llvm-cov` to
    // fail when generating coverage reports, and if there are no covfun records
    // then the covmap record isn't useful anyway.
    // This should prevent a repeat of <https://github.com/rust-lang/rust/issues/133606>.
    if covfun_records.is_empty() {
        return;
    }

    // Encode all filenames referenced by coverage mappings in this CGU.
    let filenames_buffer = global_file_table.make_filenames_buffer(tcx);
    // The `llvm-cov` tool uses this hash to associate each covfun record with
    // its corresponding filenames table, since the final binary will typically
    // contain multiple covmap records from different compilation units.
    let filenames_hash = llvm_cov::hash_bytes(&filenames_buffer);

    let mut unused_function_names = vec![];

    for covfun in &covfun_records {
        unused_function_names.extend(covfun.mangled_function_name_if_unused());

        covfun::generate_covfun_record(cx, filenames_hash, covfun)
    }

    // For unused functions, we need to take their mangled names and store them
    // in a specially-named global array. LLVM's `InstrProfiling` pass will
    // detect this global and include those names in its `__llvm_prf_names`
    // section. (See `llvm/lib/Transforms/Instrumentation/InstrProfiling.cpp`.)
    if !unused_function_names.is_empty() {
        assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

        let name_globals = unused_function_names
            .into_iter()
            .map(|mangled_function_name| cx.const_str(mangled_function_name).0)
            .collect::<Vec<_>>();
        let initializer = cx.const_array(cx.type_ptr(), &name_globals);

        let array = llvm::add_global(cx.llmod, cx.val_ty(initializer), c"__llvm_coverage_names");
        llvm::set_global_constant(array, true);
        llvm::set_linkage(array, llvm::Linkage::InternalLinkage);
        llvm::set_initializer(array, initializer);
    }

    // Generate the coverage map header, which contains the filenames used by
    // this CGU's coverage mappings, and store it in a well-known global.
    // (This is skipped if we returned early due to having no covfun records.)
    generate_covmap_record(cx, covmap_version, &filenames_buffer);
}

/// Maps "global" (per-CGU) file ID numbers to their underlying source files.
struct GlobalFileTable {
    /// This "raw" table doesn't include the working dir, so a file's
    /// global ID is its index in this set **plus one**.
    raw_file_table: FxIndexMap<StableSourceFileId, Arc<SourceFile>>,
}

impl GlobalFileTable {
    fn new() -> Self {
        Self { raw_file_table: FxIndexMap::default() }
    }

    fn global_file_id_for_file(&mut self, file: &Arc<SourceFile>) -> GlobalFileId {
        // Ensure the given file has a table entry, and get its index.
        let entry = self.raw_file_table.entry(file.stable_id);
        let raw_id = entry.index();
        entry.or_insert_with(|| Arc::clone(file));

        // The raw file table doesn't include an entry for the working dir
        // (which has ID 0), so add 1 to get the correct ID.
        GlobalFileId::from_usize(raw_id + 1)
    }

    fn make_filenames_buffer(&self, tcx: TyCtxt<'_>) -> Vec<u8> {
        let mut table = Vec::with_capacity(self.raw_file_table.len() + 1);

        // LLVM Coverage Mapping Format version 6 (zero-based encoded as 5)
        // requires setting the first filename to the compilation directory.
        // Since rustc generates coverage maps with relative paths, the
        // compilation directory can be combined with the relative paths
        // to get absolute paths, if needed.
        table.push(
            tcx.sess
                .opts
                .working_dir
                .for_scope(tcx.sess, RemapPathScopeComponents::MACRO)
                .to_string_lossy(),
        );

        // Add the regular entries after the base directory.
        table.extend(self.raw_file_table.values().map(|file| {
            file.name.for_scope(tcx.sess, RemapPathScopeComponents::MACRO).to_string_lossy()
        }));

        llvm_cov::write_filenames_to_buffer(&table)
    }
}

rustc_index::newtype_index! {
    /// An index into the CGU's overall list of file paths. The underlying paths
    /// will be embedded in the `__llvm_covmap` linker section.
    struct GlobalFileId {}
}
rustc_index::newtype_index! {
    /// An index into a function's list of global file IDs. That underlying list
    /// of local-to-global mappings will be embedded in the function's record in
    /// the `__llvm_covfun` linker section.
    struct LocalFileId {}
}

/// Holds a mapping from "local" (per-function) file IDs to "global" (per-CGU)
/// file IDs.
#[derive(Debug, Default)]
struct VirtualFileMapping {
    local_to_global: IndexVec<LocalFileId, GlobalFileId>,
    global_to_local: FxIndexMap<GlobalFileId, LocalFileId>,
}

impl VirtualFileMapping {
    fn local_id_for_global(&mut self, global_file_id: GlobalFileId) -> LocalFileId {
        *self
            .global_to_local
            .entry(global_file_id)
            .or_insert_with(|| self.local_to_global.push(global_file_id))
    }

    fn to_vec(&self) -> Vec<u32> {
        // This clone could be avoided by transmuting `&[GlobalFileId]` to `&[u32]`,
        // but it isn't hot or expensive enough to justify the extra unsafety.
        self.local_to_global.iter().map(|&global| GlobalFileId::as_u32(global)).collect()
    }
}

/// Generates the contents of the covmap record for this CGU, which mostly
/// consists of a header and a list of filenames. The record is then stored
/// as a global variable in the `__llvm_covmap` section.
fn generate_covmap_record<'ll>(cx: &CodegenCx<'ll, '_>, version: u32, filenames_buffer: &[u8]) {
    // A covmap record consists of four target-endian u32 values, followed by
    // the encoded filenames table. Two of the header fields are unused in
    // modern versions of the LLVM coverage mapping format, and are always 0.
    // <https://llvm.org/docs/CoverageMappingFormat.html#llvm-ir-representation>
    // See also `src/llvm-project/clang/lib/CodeGen/CoverageMappingGen.cpp`.
    let covmap_header = cx.const_struct(
        &[
            cx.const_u32(0), // (unused)
            cx.const_u32(filenames_buffer.len() as u32),
            cx.const_u32(0), // (unused)
            cx.const_u32(version),
        ],
        /* packed */ false,
    );
    let covmap_record = cx
        .const_struct(&[covmap_header, cx.const_bytes(filenames_buffer)], /* packed */ false);

    let covmap_global =
        llvm::add_global(cx.llmod, cx.val_ty(covmap_record), &llvm_cov::covmap_var_name());
    llvm::set_initializer(covmap_global, covmap_record);
    llvm::set_global_constant(covmap_global, true);
    llvm::set_linkage(covmap_global, llvm::Linkage::PrivateLinkage);
    llvm::set_section(covmap_global, &llvm_cov::covmap_section_name(cx.llmod));
    // LLVM's coverage mapping format specifies 8-byte alignment for items in this section.
    // <https://llvm.org/docs/CoverageMappingFormat.html>
    llvm::set_alignment(covmap_global, Align::EIGHT);

    cx.add_used_global(covmap_global);
}

/// Each CGU will normally only emit coverage metadata for the functions that it actually generates.
/// But since we don't want unused functions to disappear from coverage reports, we also scan for
/// functions that were instrumented but are not participating in codegen.
///
/// These unused functions don't need to be codegenned, but we do need to add them to the function
/// coverage map (in a single designated CGU) so that we still emit coverage mappings for them.
/// We also end up adding their symbol names to a special global array that LLVM will include in
/// its embedded coverage data.
fn gather_unused_function_instances<'tcx>(cx: &CodegenCx<'_, 'tcx>) -> Vec<ty::Instance<'tcx>> {
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
