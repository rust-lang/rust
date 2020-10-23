use crate::common::CodegenCx;
use crate::coverageinfo;
use crate::llvm;

use llvm::coverageinfo::CounterMappingRegion;
use rustc_codegen_ssa::coverageinfo::map::{Counter, CounterExpression};
use rustc_codegen_ssa::traits::{BaseTypeMethods, ConstMethods};
use rustc_data_structures::fx::FxIndexSet;
use rustc_llvm::RustString;
use rustc_middle::mir::coverage::CodeRegion;

use std::ffi::CString;

use tracing::debug;

/// Generates and exports the Coverage Map.
///
/// This Coverage Map complies with Coverage Mapping Format version 3 (zero-based encoded as 2),
/// as defined at [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/llvmorg-8.0.0/llvm/docs/CoverageMappingFormat.rst#llvm-code-coverage-mapping-format)
/// and published in Rust's current (July 2020) fork of LLVM. This version is supported by the
/// LLVM coverage tools (`llvm-profdata` and `llvm-cov`) bundled with Rust's fork of LLVM.
///
/// Consequently, Rust's bundled version of Clang also generates Coverage Maps compliant with
/// version 3. Clang's implementation of Coverage Map generation was referenced when implementing
/// this Rust version, and though the format documentation is very explicit and detailed, some
/// undocumented details in Clang's implementation (that may or may not be important) were also
/// replicated for Rust's Coverage Map.
pub fn finalize<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>) {
    let function_coverage_map = match cx.coverage_context() {
        Some(ctx) => ctx.take_function_coverage_map(),
        None => return,
    };
    if function_coverage_map.is_empty() {
        // This module has no functions with coverage instrumentation
        return;
    }

    let mut mapgen = CoverageMapGenerator::new();

    // Encode coverage mappings and generate function records
    let mut function_records = Vec::<&'ll llvm::Value>::new();
    let coverage_mappings_buffer = llvm::build_byte_buffer(|coverage_mappings_buffer| {
        for (instance, function_coverage) in function_coverage_map.into_iter() {
            debug!("Generate coverage map for: {:?}", instance);

            let mangled_function_name = cx.tcx.symbol_name(instance).to_string();
            let function_source_hash = function_coverage.source_hash();
            let (expressions, counter_regions) =
                function_coverage.get_expressions_and_counter_regions();

            let old_len = coverage_mappings_buffer.len();
            mapgen.write_coverage_mappings(expressions, counter_regions, coverage_mappings_buffer);
            let mapping_data_size = coverage_mappings_buffer.len() - old_len;
            debug_assert!(
                mapping_data_size > 0,
                "Every `FunctionCoverage` should have at least one counter"
            );

            let function_record = mapgen.make_function_record(
                cx,
                mangled_function_name,
                function_source_hash,
                mapping_data_size,
            );
            function_records.push(function_record);
        }
    });

    // Encode all filenames referenced by counters/expressions in this module
    let filenames_buffer = llvm::build_byte_buffer(|filenames_buffer| {
        coverageinfo::write_filenames_section_to_buffer(&mapgen.filenames, filenames_buffer);
    });

    // Generate the LLVM IR representation of the coverage map and store it in a well-known global
    mapgen.save_generated_coverage_map(
        cx,
        function_records,
        filenames_buffer,
        coverage_mappings_buffer,
    );
}

struct CoverageMapGenerator {
    filenames: FxIndexSet<CString>,
}

impl CoverageMapGenerator {
    fn new() -> Self {
        Self { filenames: FxIndexSet::default() }
    }

    /// Using the `expressions` and `counter_regions` collected for the current function, generate
    /// the `mapping_regions` and `virtual_file_mapping`, and capture any new filenames. Then use
    /// LLVM APIs to encode the `virtual_file_mapping`, `expressions`, and `mapping_regions` into
    /// the given `coverage_mappings` byte buffer, compliant with the LLVM Coverage Mapping format.
    fn write_coverage_mappings(
        &mut self,
        expressions: Vec<CounterExpression>,
        counter_regions: impl Iterator<Item = (Counter, &'a CodeRegion)>,
        coverage_mappings_buffer: &RustString,
    ) {
        let mut counter_regions = counter_regions.collect::<Vec<_>>();
        if counter_regions.is_empty() {
            return;
        }

        let mut virtual_file_mapping = Vec::new();
        let mut mapping_regions = Vec::new();
        let mut current_file_name = None;
        let mut current_file_id = 0;

        // Convert the list of (Counter, CodeRegion) pairs to an array of `CounterMappingRegion`, sorted
        // by filename and position. Capture any new files to compute the `CounterMappingRegion`s
        // `file_id` (indexing files referenced by the current function), and construct the
        // function-specific `virtual_file_mapping` from `file_id` to its index in the module's
        // `filenames` array.
        counter_regions.sort_unstable_by_key(|(_counter, region)| *region);
        for (counter, region) in counter_regions {
            let CodeRegion { file_name, start_line, start_col, end_line, end_col } = *region;
            let same_file = current_file_name.as_ref().map_or(false, |p| *p == file_name);
            if !same_file {
                if current_file_name.is_some() {
                    current_file_id += 1;
                }
                current_file_name = Some(file_name);
                let c_filename = CString::new(file_name.to_string())
                    .expect("null error converting filename to C string");
                debug!("  file_id: {} = '{:?}'", current_file_id, c_filename);
                let (filenames_index, _) = self.filenames.insert_full(c_filename);
                virtual_file_mapping.push(filenames_index as u32);
            }
            debug!("Adding counter {:?} to map for {:?}", counter, region,);
            mapping_regions.push(CounterMappingRegion::code_region(
                counter,
                current_file_id,
                start_line,
                start_col,
                end_line,
                end_col,
            ));
        }

        // Encode and append the current function's coverage mapping data
        coverageinfo::write_mapping_to_buffer(
            virtual_file_mapping,
            expressions,
            mapping_regions,
            coverage_mappings_buffer,
        );
    }

    /// Generate and return the function record `Value`
    fn make_function_record(
        &mut self,
        cx: &CodegenCx<'ll, 'tcx>,
        mangled_function_name: String,
        function_source_hash: u64,
        mapping_data_size: usize,
    ) -> &'ll llvm::Value {
        let name_ref = coverageinfo::compute_hash(&mangled_function_name);
        let name_ref_val = cx.const_u64(name_ref);
        let mapping_data_size_val = cx.const_u32(mapping_data_size as u32);
        let func_hash_val = cx.const_u64(function_source_hash);
        cx.const_struct(
            &[name_ref_val, mapping_data_size_val, func_hash_val],
            /*packed=*/ true,
        )
    }

    /// Combine the filenames and coverage mappings buffers, construct coverage map header and the
    /// array of function records, and combine everything into the complete coverage map. Save the
    /// coverage map data into the LLVM IR as a static global using a specific, well-known section
    /// and name.
    fn save_generated_coverage_map(
        self,
        cx: &CodegenCx<'ll, 'tcx>,
        function_records: Vec<&'ll llvm::Value>,
        filenames_buffer: Vec<u8>,
        mut coverage_mappings_buffer: Vec<u8>,
    ) {
        // Concatenate the encoded filenames and encoded coverage mappings, and add additional zero
        // bytes as-needed to ensure 8-byte alignment.
        let mut coverage_size = coverage_mappings_buffer.len();
        let filenames_size = filenames_buffer.len();
        let remaining_bytes =
            (filenames_size + coverage_size) % coverageinfo::COVMAP_VAR_ALIGN_BYTES;
        if remaining_bytes > 0 {
            let pad = coverageinfo::COVMAP_VAR_ALIGN_BYTES - remaining_bytes;
            coverage_mappings_buffer.append(&mut [0].repeat(pad));
            coverage_size += pad;
        }
        let filenames_and_coverage_mappings = [filenames_buffer, coverage_mappings_buffer].concat();
        let filenames_and_coverage_mappings_val =
            cx.const_bytes(&filenames_and_coverage_mappings[..]);

        debug!(
            "cov map: n_records = {}, filenames_size = {}, coverage_size = {}, 0-based version = {}",
            function_records.len(),
            filenames_size,
            coverage_size,
            coverageinfo::mapping_version()
        );

        // Create the coverage data header
        let n_records_val = cx.const_u32(function_records.len() as u32);
        let filenames_size_val = cx.const_u32(filenames_size as u32);
        let coverage_size_val = cx.const_u32(coverage_size as u32);
        let version_val = cx.const_u32(coverageinfo::mapping_version());
        let cov_data_header_val = cx.const_struct(
            &[n_records_val, filenames_size_val, coverage_size_val, version_val],
            /*packed=*/ false,
        );

        // Create the function records array
        let name_ref_from_u64 = cx.type_i64();
        let mapping_data_size_from_u32 = cx.type_i32();
        let func_hash_from_u64 = cx.type_i64();
        let function_record_ty = cx.type_struct(
            &[name_ref_from_u64, mapping_data_size_from_u32, func_hash_from_u64],
            /*packed=*/ true,
        );
        let function_records_val = cx.const_array(function_record_ty, &function_records[..]);

        // Create the complete LLVM coverage data value to add to the LLVM IR
        let cov_data_val = cx.const_struct(
            &[cov_data_header_val, function_records_val, filenames_and_coverage_mappings_val],
            /*packed=*/ false,
        );

        // Save the coverage data value to LLVM IR
        coverageinfo::save_map_to_mod(cx, cov_data_val);
    }
}
