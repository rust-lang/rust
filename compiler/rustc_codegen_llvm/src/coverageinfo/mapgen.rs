use crate::common::CodegenCx;
use crate::coverageinfo;
use crate::coverageinfo::ffi::CounterMappingRegion;
use crate::coverageinfo::map_data::{FunctionCoverage, FunctionCoverageCollector};
use crate::llvm;

use itertools::Itertools as _;
use rustc_codegen_ssa::traits::{BaseTypeMethods, ConstMethods};
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::mir;
use rustc_middle::mir::coverage::CodeRegion;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::DefIdSet;
use rustc_span::Symbol;

/// Generates and exports the Coverage Map.
///
/// Rust Coverage Map generation supports LLVM Coverage Mapping Format version
/// 6 (zero-based encoded as 5), as defined at
/// [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/rustc/13.0-2021-09-30/llvm/docs/CoverageMappingFormat.rst#llvm-code-coverage-mapping-format).
/// These versions are supported by the LLVM coverage tools (`llvm-profdata` and `llvm-cov`)
/// bundled with Rust's fork of LLVM.
///
/// Consequently, Rust's bundled version of Clang also generates Coverage Maps compliant with
/// the same version. Clang's implementation of Coverage Map generation was referenced when
/// implementing this Rust version, and though the format documentation is very explicit and
/// detailed, some undocumented details in Clang's implementation (that may or may not be important)
/// were also replicated for Rust's Coverage Map.
pub fn finalize(cx: &CodegenCx<'_, '_>) {
    let tcx = cx.tcx;

    // Ensure the installed version of LLVM supports Coverage Map Version 6
    // (encoded as a zero-based value: 5), which was introduced with LLVM 13.
    let version = coverageinfo::mapping_version();
    assert_eq!(version, 5, "The `CoverageMappingVersion` exposed by `llvm-wrapper` is out of sync");

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
    let cov_data_val = generate_coverage_map(cx, version, filenames_size, filenames_val);
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
        let working_dir: &str = &tcx.sess.opts.working_dir.for_codegen(&tcx.sess).to_string_lossy();

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
    // Tell the newtype macro to not generate `Encode`/`Decode` impls.
    #[custom_encodable]
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
        &counter_regions.group_by(|(_counter, region)| region.file_name)
    {
        // Look up the global file ID for this filename.
        let global_file_id = global_file_table.global_file_id_for_file_name(file_name);

        // Associate that global file ID with a local file ID for this function.
        let local_file_id = virtual_file_mapping.local_id_for_global(global_file_id);
        debug!("  file id: {local_file_id:?} => global {global_file_id} = '{file_name:?}'");

        // For each counter/region pair in this function+file, convert it to a
        // form suitable for FFI.
        for (counter, region) in counter_regions_for_file {
            let CodeRegion { file_name: _, start_line, start_col, end_line, end_col } = *region;

            debug!("Adding counter {counter:?} to map for {region:?}");
            mapping_regions.push(CounterMappingRegion::code_region(
                counter,
                local_file_id.as_u32(),
                start_line,
                start_col,
                end_line,
                end_col,
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

/// When finalizing the coverage map, `FunctionCoverage` only has the `CodeRegion`s and counters for
/// the functions that went through codegen; such as public functions and "used" functions
/// (functions referenced by other "used" or public items). Any other functions considered unused,
/// or "Unreachable", were still parsed and processed through the MIR stage, but were not
/// codegenned. (Note that `-Clink-dead-code` can force some unused code to be codegenned, but
/// that flag is known to cause other errors, when combined with `-C instrument-coverage`; and
/// `-Clink-dead-code` will not generate code for unused generic functions.)
///
/// We can find the unused functions (including generic functions) by the set difference of all MIR
/// `DefId`s (`tcx` query `mir_keys`) minus the codegenned `DefId`s (`codegenned_and_inlined_items`).
///
/// These unused functions don't need to be codegenned, but we do need to add them to the function
/// coverage map (in a single designated CGU) so that we still emit coverage mappings for them.
/// We also end up adding their symbol names to a special global array that LLVM will include in
/// its embedded coverage data.
fn add_unused_functions(cx: &CodegenCx<'_, '_>) {
    assert!(cx.codegen_unit.is_code_coverage_dead_code_cgu());

    let tcx = cx.tcx;

    let ignore_unused_generics = tcx.sess.instrument_coverage_except_unused_generics();

    let eligible_def_ids: Vec<DefId> = tcx
        .mir_keys(())
        .iter()
        .filter_map(|local_def_id| {
            let def_id = local_def_id.to_def_id();
            let kind = tcx.def_kind(def_id);
            // `mir_keys` will give us `DefId`s for all kinds of things, not
            // just "functions", like consts, statics, etc. Filter those out.
            // If `ignore_unused_generics` was specified, filter out any
            // generic functions from consideration as well.
            if !matches!(
                kind,
                DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Coroutine
            ) {
                return None;
            }
            if ignore_unused_generics && tcx.generics_of(def_id).requires_monomorphization(tcx) {
                return None;
            }
            Some(local_def_id.to_def_id())
        })
        .collect();

    let codegenned_def_ids = codegenned_and_inlined_items(tcx);

    // For each `DefId` that should have coverage instrumentation but wasn't
    // codegenned, add it to the function coverage map as an unused function.
    for def_id in eligible_def_ids.into_iter().filter(|id| !codegenned_def_ids.contains(id)) {
        // Skip any function that didn't have coverage data added to it by the
        // coverage instrumentor.
        let body = tcx.instance_mir(ty::InstanceDef::Item(def_id));
        let Some(function_coverage_info) = body.function_coverage_info.as_deref() else {
            continue;
        };

        debug!("generating unused fn: {def_id:?}");
        let instance = declare_unused_fn(tcx, def_id);
        add_unused_function_coverage(cx, instance, function_coverage_info);
    }
}

/// All items participating in code generation together with (instrumented)
/// items inlined into them.
fn codegenned_and_inlined_items(tcx: TyCtxt<'_>) -> DefIdSet {
    let (items, cgus) = tcx.collect_and_partition_mono_items(());
    let mut visited = DefIdSet::default();
    let mut result = items.clone();

    for cgu in cgus {
        for item in cgu.items().keys() {
            if let mir::mono::MonoItem::Fn(ref instance) = item {
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

    result
}

fn declare_unused_fn<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> ty::Instance<'tcx> {
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

fn add_unused_function_coverage<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    instance: ty::Instance<'tcx>,
    function_coverage_info: &'tcx mir::coverage::FunctionCoverageInfo,
) {
    // An unused function's mappings will automatically be rewritten to map to
    // zero, because none of its counters/expressions are marked as seen.
    let function_coverage = FunctionCoverageCollector::unused(instance, function_coverage_info);

    if let Some(coverage_context) = cx.coverage_context() {
        coverage_context.function_coverage_map.borrow_mut().insert(instance, function_coverage);
    } else {
        bug!("Could not get the `coverage_context`");
    }
}
