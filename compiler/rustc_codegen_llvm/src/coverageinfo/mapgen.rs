use crate::common::CodegenCx;
use crate::coverageinfo;
use crate::coverageinfo::ffi::CounterMappingRegion;
use crate::coverageinfo::map_data::FunctionCoverage;
use crate::coverageinfo::unused;
use crate::llvm;

use rustc_codegen_ssa::traits::ConstMethods;
use rustc_data_structures::fx::FxIndexSet;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::mir::coverage::CodeRegion;
use rustc_middle::ty::TyCtxt;
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
        unused::add_unused_functions(cx);
    }

    let function_coverage_map = match cx.coverage_context() {
        Some(ctx) => ctx.take_function_coverage_map(),
        None => return,
    };

    if function_coverage_map.is_empty() {
        // This module has no functions with coverage instrumentation
        return;
    }

    let mut global_file_table = GlobalFileTable::new(tcx);

    // Encode coverage mappings and generate function records
    let mut function_data = Vec::new();
    for (instance, function_coverage) in function_coverage_map {
        debug!("Generate function coverage for {}, {:?}", cx.codegen_unit.name(), instance);
        let mangled_function_name = tcx.symbol_name(instance).name;
        let source_hash = function_coverage.source_hash();
        let is_used = function_coverage.is_used();

        let coverage_mapping_buffer =
            encode_mappings_for_function(&mut global_file_table, &function_coverage);

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

        function_data.push((mangled_function_name, source_hash, is_used, coverage_mapping_buffer));
    }

    // Encode all filenames referenced by counters/expressions in this module
    let filenames_buffer = global_file_table.into_filenames_buffer();

    let filenames_size = filenames_buffer.len();
    let filenames_val = cx.const_bytes(&filenames_buffer);
    let filenames_ref = coverageinfo::hash_bytes(&filenames_buffer);

    // Generate the LLVM IR representation of the coverage map and store it in a well-known global
    let cov_data_val = generate_coverage_map(cx, version, filenames_size, filenames_val);

    let covfun_section_name = coverageinfo::covfun_section_name(cx);
    for (mangled_function_name, source_hash, is_used, coverage_mapping_buffer) in function_data {
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

    // Save the coverage data value to LLVM IR
    coverageinfo::save_cov_data_to_mod(cx, cov_data_val);
}

struct GlobalFileTable {
    global_file_table: FxIndexSet<Symbol>,
}

impl GlobalFileTable {
    fn new(tcx: TyCtxt<'_>) -> Self {
        let mut global_file_table = FxIndexSet::default();
        // LLVM Coverage Mapping Format version 6 (zero-based encoded as 5)
        // requires setting the first filename to the compilation directory.
        // Since rustc generates coverage maps with relative paths, the
        // compilation directory can be combined with the relative paths
        // to get absolute paths, if needed.
        let working_dir = Symbol::intern(
            &tcx.sess.opts.working_dir.remapped_path_if_available().to_string_lossy(),
        );
        global_file_table.insert(working_dir);
        Self { global_file_table }
    }

    fn global_file_id_for_file_name(&mut self, file_name: Symbol) -> u32 {
        let (global_file_id, _) = self.global_file_table.insert_full(file_name);
        global_file_id as u32
    }

    fn into_filenames_buffer(self) -> Vec<u8> {
        // This method takes `self` so that the caller can't accidentally
        // modify the original file table after encoding it into a buffer.

        llvm::build_byte_buffer(|buffer| {
            coverageinfo::write_filenames_section_to_buffer(
                self.global_file_table.iter().map(Symbol::as_str),
                buffer,
            );
        })
    }
}

/// Using the expressions and counter regions collected for a single function,
/// generate the variable-sized payload of its corresponding `__llvm_covfun`
/// entry. The payload is returned as a vector of bytes.
///
/// Newly-encountered filenames will be added to the global file table.
fn encode_mappings_for_function(
    global_file_table: &mut GlobalFileTable,
    function_coverage: &FunctionCoverage<'_>,
) -> Vec<u8> {
    let (expressions, counter_regions) = function_coverage.get_expressions_and_counter_regions();

    let mut counter_regions = counter_regions.collect::<Vec<_>>();
    if counter_regions.is_empty() {
        return Vec::new();
    }

    let mut virtual_file_mapping = IndexVec::<u32, u32>::new();
    let mut mapping_regions = Vec::with_capacity(counter_regions.len());

    // Sort the list of (counter, region) mapping pairs by region, so that they
    // can be grouped by filename. Prepare file IDs for each filename, and
    // prepare the mapping data so that we can pass it through FFI to LLVM.
    counter_regions.sort_by_key(|(_counter, region)| *region);
    for counter_regions_for_file in
        counter_regions.group_by(|(_, a), (_, b)| a.file_name == b.file_name)
    {
        // Look up (or allocate) the global file ID for this filename.
        let file_name = counter_regions_for_file[0].1.file_name;
        let global_file_id = global_file_table.global_file_id_for_file_name(file_name);

        // Associate that global file ID with a local file ID for this function.
        let local_file_id: u32 = virtual_file_mapping.push(global_file_id);
        debug!("  file id: local {local_file_id} => global {global_file_id} = '{file_name:?}'");

        // For each counter/region pair in this function+file, convert it to a
        // form suitable for FFI.
        for &(counter, region) in counter_regions_for_file {
            let CodeRegion { file_name: _, start_line, start_col, end_line, end_col } = *region;

            debug!("Adding counter {counter:?} to map for {region:?}");
            mapping_regions.push(CounterMappingRegion::code_region(
                counter,
                local_file_id,
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
            virtual_file_mapping.raw,
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
