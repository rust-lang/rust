use itertools::Itertools as _;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, ConstCodegenMethods};
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::IndexVec;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::Symbol;
use rustc_span::def_id::DefIdSet;
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::ffi::CounterMappingRegion;
use crate::coverageinfo::map_data::{FunctionCoverage, FunctionCoverageCollector};
use crate::{coverageinfo, llvm};

/// Generates and exports the Coverage Map.
///
/// Rust Coverage Map generation supports LLVM Coverage Mapping Format versions
/// 6 and 7 (encoded as 5 and 6 respectively), as described at
/// [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/rustc/18.0-2024-02-13/llvm/docs/CoverageMappingFormat.rst).
/// These versions are supported by the LLVM coverage tools (`llvm-profdata` and `llvm-cov`)
/// distributed in the `llvm-tools-preview` rustup component.
///
/// Consequently, Rust's bundled version of Clang also generates Coverage Maps compliant with
/// the same version. Clang's implementation of Coverage Map generation was referenced when
/// implementing this Rust version, and though the format documentation is very explicit and
/// detailed, some undocumented details in Clang's implementation (that may or may not be important)
/// were also replicated for Rust's Coverage Map.
pub(crate) fn finalize(cx: &CodegenCx<'_, '_>) {
    let tcx = cx.tcx;

    // Ensure that LLVM is using a version of the coverage mapping format that
    // agrees with our Rust-side code. Expected versions (encoded as n-1) are:
    // - `CovMapVersion::Version6` (5) used by LLVM 13-17
    // - `CovMapVersion::Version7` (6) used by LLVM 18
    let covmap_version = {
        let llvm_covmap_version = coverageinfo::mapping_version();
        let expected_versions = 5..=6;
        assert!(
            expected_versions.contains(&llvm_covmap_version),
            "Coverage mapping version exposed by `llvm-wrapper` is out of sync; \
            expected {expected_versions:?} but was {llvm_covmap_version}"
        );
        // This is the version number that we will embed in the covmap section:
        llvm_covmap_version
    };

    debug!("Generating coverage map for CodegenUnit: `{}`", cx.codegen_unit.name());

    // In order to show that unused functions have coverage counts of zero (0), LLVM requires the
    // functions exist. Generate synthetic functions with a (required) single counter, and add the
    // MIR `Coverage` code regions to the `function_coverage_map`, before calling
    // `ctx.take_function_coverage_map()`.
    if cx.codegen_unit.is_code_coverage_dead_code_cgu() {
        add_unused_functions(cx);
    }

    let function_coverage_map = match cx.coverage_context() {
        Some(ctx) => ctx.take_function_coverage_map(),
        None => return,
    };

    if function_coverage_map.is_empty() {
        // This module has no functions with coverage instrumentation
        return;
    }

    let function_coverage_entries = function_coverage_map
        .into_iter()
        .map(|(instance, function_coverage)| (instance, function_coverage.into_finished()))
        .collect::<Vec<_>>();

    let all_file_names =
        function_coverage_entries.iter().flat_map(|(_, fn_cov)| fn_cov.all_file_names());
    let global_file_table = GlobalFileTable::new(all_file_names);

    // Encode all filenames referenced by coverage mappings in this CGU.
    let filenames_buffer = global_file_table.make_filenames_buffer(tcx);

    let filenames_size = filenames_buffer.len();
    let filenames_val = cx.const_bytes(&filenames_buffer);
    let filenames_ref = coverageinfo::hash_bytes(&filenames_buffer);

    // Generate the coverage map header, which contains the filenames used by
    // this CGU's coverage mappings, and store it in a well-known global.
    let cov_data_val = generate_coverage_map(cx, covmap_version, filenames_size, filenames_val);
    coverageinfo::save_cov_data_to_mod(cx, cov_data_val);

    let mut unused_function_names = Vec::new();
    let covfun_section_name = coverageinfo::covfun_section_name(cx);

    // Encode coverage mappings and generate function records
    for (instance, function_coverage) in function_coverage_entries {
        debug!("Generate function coverage for {}, {:?}", cx.codegen_unit.name(), instance);

        let mangled_function_name = tcx.symbol_name(instance).name;
        let source_hash = function_coverage.source_hash();
        let is_used = function_coverage.is_used();

        let coverage_mapping_buffer =
            encode_mappings_for_function(&global_file_table, &function_coverage);

        if coverage_mapping_buffer.is_empty() {
            if function_coverage.is_used() {
                bug!(
                    "A used function should have had coverage mapping data but did not: {}",
                    mangled_function_name
                );
            } else {
                debug!("unused function had no coverage mapping data: {}", mangled_function_name);
                continue;
            }
        }

        if !is_used {
            unused_function_names.push(mangled_function_name);
        }

        save_function_record(
            cx,
            &covfun_section_name,
            mangled_function_name,
            source_hash,
            filenames_ref,
            coverage_mapping_buffer,
            is_used,
        );
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

        let array = llvm::add_global(cx.llmod, cx.val_ty(initializer), "__llvm_coverage_names");
        llvm::set_global_constant(array, true);
        llvm::set_linkage(array, llvm::Linkage::InternalLinkage);
        llvm::set_initializer(array, initializer);
    }
}

/// Maps "global" (per-CGU) file ID numbers to their underlying filenames.
struct GlobalFileTable {
    /// This "raw" table doesn't include the working dir, so a filename's
    /// global ID is its index in this set **plus one**.
    raw_file_table: FxIndexSet<Symbol>,
}

impl GlobalFileTable {
    fn new(all_file_names: impl IntoIterator<Item = Symbol>) -> Self {
        // Collect all of the filenames into a set. Filenames usually come in
        // contiguous runs, so we can dedup adjacent ones to save work.
        let mut raw_file_table = all_file_names.into_iter().dedup().collect::<FxIndexSet<Symbol>>();

        // Sort the file table by its actual string values, not the arbitrary
        // ordering of its symbols.
        raw_file_table.sort_unstable_by(|a, b| a.as_str().cmp(b.as_str()));

        Self { raw_file_table }
    }

    fn global_file_id_for_file_name(&self, file_name: Symbol) -> u32 {
        let raw_id = self.raw_file_table.get_index_of(&file_name).unwrap_or_else(|| {
            bug!("file name not found in prepared global file table: {file_name}");
        });
        // The raw file table doesn't include an entry for the working dir
        // (which has ID 0), so add 1 to get the correct ID.
        (raw_id + 1) as u32
    }

    fn make_filenames_buffer(&self, tcx: TyCtxt<'_>) -> Vec<u8> {
        // LLVM Coverage Mapping Format version 6 (zero-based encoded as 5)
        // requires setting the first filename to the compilation directory.
        // Since rustc generates coverage maps with relative paths, the
        // compilation directory can be combined with the relative paths
        // to get absolute paths, if needed.
        use rustc_session::RemapFileNameExt;
        use rustc_session::config::RemapPathScopeComponents;
        let working_dir: &str = &tcx
            .sess
            .opts
            .working_dir
            .for_scope(tcx.sess, RemapPathScopeComponents::MACRO)
            .to_string_lossy();

        llvm::build_byte_buffer(|buffer| {
            coverageinfo::write_filenames_section_to_buffer(
                // Insert the working dir at index 0, before the other filenames.
                std::iter::once(working_dir).chain(self.raw_file_table.iter().map(Symbol::as_str)),
                buffer,
            );
        })
    }
}

rustc_index::newtype_index! {
    struct LocalFileId {}
}

/// Holds a mapping from "local" (per-function) file IDs to "global" (per-CGU)
/// file IDs.
#[derive(Default)]
struct VirtualFileMapping {
    local_to_global: IndexVec<LocalFileId, u32>,
    global_to_local: FxIndexMap<u32, LocalFileId>,
}

impl VirtualFileMapping {
    fn local_id_for_global(&mut self, global_file_id: u32) -> LocalFileId {
        *self
            .global_to_local
            .entry(global_file_id)
            .or_insert_with(|| self.local_to_global.push(global_file_id))
    }

    fn into_vec(self) -> Vec<u32> {
        self.local_to_global.raw
    }
}

/// Using the expressions and counter regions collected for a single function,
/// generate the variable-sized payload of its corresponding `__llvm_covfun`
/// entry. The payload is returned as a vector of bytes.
///
/// Newly-encountered filenames will be added to the global file table.
fn encode_mappings_for_function(
    global_file_table: &GlobalFileTable,
    function_coverage: &FunctionCoverage<'_>,
) -> Vec<u8> {
    let counter_regions = function_coverage.counter_regions();
    if counter_regions.is_empty() {
        return Vec::new();
    }

    let expressions = function_coverage.counter_expressions().collect::<Vec<_>>();

    let mut virtual_file_mapping = VirtualFileMapping::default();
    let mut mapping_regions = Vec::with_capacity(counter_regions.len());

    // Group mappings into runs with the same filename, preserving the order
    // yielded by `FunctionCoverage`.
    // Prepare file IDs for each filename, and prepare the mapping data so that
    // we can pass it through FFI to LLVM.
    for (file_name, counter_regions_for_file) in
        &counter_regions.group_by(|(_, region)| region.file_name)
    {
        // Look up the global file ID for this filename.
        let global_file_id = global_file_table.global_file_id_for_file_name(file_name);

        // Associate that global file ID with a local file ID for this function.
        let local_file_id = virtual_file_mapping.local_id_for_global(global_file_id);
        debug!("  file id: {local_file_id:?} => global {global_file_id} = '{file_name:?}'");

        // For each counter/region pair in this function+file, convert it to a
        // form suitable for FFI.
        for (mapping_kind, region) in counter_regions_for_file {
            debug!("Adding counter {mapping_kind:?} to map for {region:?}");
            mapping_regions.push(CounterMappingRegion::from_mapping(
                &mapping_kind,
                local_file_id.as_u32(),
                region,
            ));
        }
    }

    // Encode the function's coverage mappings into a buffer.
    llvm::build_byte_buffer(|buffer| {
        coverageinfo::write_mapping_to_buffer(
            virtual_file_mapping.into_vec(),
            expressions,
            mapping_regions,
            buffer,
        );
    })
}

/// Construct coverage map header and the array of function records, and combine them into the
/// coverage map. Save the coverage map data into the LLVM IR as a static global using a
/// specific, well-known section and name.
fn generate_coverage_map<'ll>(
    cx: &CodegenCx<'ll, '_>,
    version: u32,
    filenames_size: usize,
    filenames_val: &'ll llvm::Value,
) -> &'ll llvm::Value {
    debug!("cov map: filenames_size = {}, 0-based version = {}", filenames_size, version);

    // Create the coverage data header (Note, fields 0 and 2 are now always zero,
    // as of `llvm::coverage::CovMapVersion::Version4`.)
    let zero_was_n_records_val = cx.const_u32(0);
    let filenames_size_val = cx.const_u32(filenames_size as u32);
    let zero_was_coverage_size_val = cx.const_u32(0);
    let version_val = cx.const_u32(version);
    let cov_data_header_val = cx.const_struct(
        &[zero_was_n_records_val, filenames_size_val, zero_was_coverage_size_val, version_val],
        /*packed=*/ false,
    );

    // Create the complete LLVM coverage data value to add to the LLVM IR
    cx.const_struct(&[cov_data_header_val, filenames_val], /*packed=*/ false)
}

/// Construct a function record and combine it with the function's coverage mapping data.
/// Save the function record into the LLVM IR as a static global using a
/// specific, well-known section and name.
fn save_function_record(
    cx: &CodegenCx<'_, '_>,
    covfun_section_name: &str,
    mangled_function_name: &str,
    source_hash: u64,
    filenames_ref: u64,
    coverage_mapping_buffer: Vec<u8>,
    is_used: bool,
) {
    // Concatenate the encoded coverage mappings
    let coverage_mapping_size = coverage_mapping_buffer.len();
    let coverage_mapping_val = cx.const_bytes(&coverage_mapping_buffer);

    let func_name_hash = coverageinfo::hash_bytes(mangled_function_name.as_bytes());
    let func_name_hash_val = cx.const_u64(func_name_hash);
    let coverage_mapping_size_val = cx.const_u32(coverage_mapping_size as u32);
    let source_hash_val = cx.const_u64(source_hash);
    let filenames_ref_val = cx.const_u64(filenames_ref);
    let func_record_val = cx.const_struct(
        &[
            func_name_hash_val,
            coverage_mapping_size_val,
            source_hash_val,
            filenames_ref_val,
            coverage_mapping_val,
        ],
        /*packed=*/ true,
    );

    coverageinfo::save_func_record_to_mod(
        cx,
        covfun_section_name,
        func_name_hash,
        func_record_val,
        is_used,
    );
}

/// Each CGU will normally only emit coverage metadata for the functions that it actually generates.
/// But since we don't want unused functions to disappear from coverage reports, we also scan for
/// functions that were instrumented but are not participating in codegen.
///
/// These unused functions don't need to be codegenned, but we do need to add them to the function
/// coverage map (in a single designated CGU) so that we still emit coverage mappings for them.
/// We also end up adding their symbol names to a special global array that LLVM will include in
/// its embedded coverage data.
fn add_unused_functions(cx: &CodegenCx<'_, '_>) {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let tcx = cx.tcx;
    let usage = prepare_usage_sets(tcx);

    let is_unused_fn = |def_id: LocalDefId| -> bool {
        let def_id = def_id.to_def_id();

        // To be eligible for "unused function" mappings, a definition must:
        // - Be function-like
        // - Not participate directly in codegen (or have lost all its coverage statements)
        // - Not have any coverage statements inlined into codegenned functions
        tcx.def_kind(def_id).is_fn_like()
            && (!usage.all_mono_items.contains(&def_id)
                || usage.missing_own_coverage.contains(&def_id))
            && !usage.used_via_inlining.contains(&def_id)
    };

    // Scan for unused functions that were instrumented for coverage.
    for def_id in tcx.mir_keys(()).iter().copied().filter(|&def_id| is_unused_fn(def_id)) {
        // Get the coverage info from MIR, skipping functions that were never instrumented.
        let body = tcx.optimized_mir(def_id);
        let Some(function_coverage_info) = body.function_coverage_info.as_deref() else { continue };

        // FIXME(79651): Consider trying to filter out dummy instantiations of
        // unused generic functions from library crates, because they can produce
        // "unused instantiation" in coverage reports even when they are actually
        // used by some downstream crate in the same binary.

        debug!("generating unused fn: {def_id:?}");
        add_unused_function_coverage(cx, def_id, function_coverage_info);
    }
}

struct UsageSets<'tcx> {
    all_mono_items: &'tcx DefIdSet,
    used_via_inlining: FxHashSet<DefId>,
    missing_own_coverage: FxHashSet<DefId>,
}

/// Prepare sets of definitions that are relevant to deciding whether something
/// is an "unused function" for coverage purposes.
fn prepare_usage_sets<'tcx>(tcx: TyCtxt<'tcx>) -> UsageSets<'tcx> {
    let (all_mono_items, cgus) = tcx.collect_and_partition_mono_items(());

    // Obtain a MIR body for each function participating in codegen, via an
    // arbitrary instance.
    let mut def_ids_seen = FxHashSet::default();
    let def_and_mir_for_all_mono_fns = cgus
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

fn add_unused_function_coverage<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    def_id: LocalDefId,
    function_coverage_info: &'tcx mir::coverage::FunctionCoverageInfo,
) {
    let tcx = cx.tcx;
    let def_id = def_id.to_def_id();

    // Make a dummy instance that fills in all generics with placeholders.
    let instance = ty::Instance::new(
        def_id,
        ty::GenericArgs::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    );

    // An unused function's mappings will automatically be rewritten to map to
    // zero, because none of its counters/expressions are marked as seen.
    let function_coverage = FunctionCoverageCollector::unused(instance, function_coverage_info);

    if let Some(coverage_context) = cx.coverage_context() {
        coverage_context.function_coverage_map.borrow_mut().insert(instance, function_coverage);
    } else {
        bug!("Could not get the `coverage_context`");
    }
}
