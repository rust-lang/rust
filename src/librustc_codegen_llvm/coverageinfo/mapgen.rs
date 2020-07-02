use crate::llvm;

use crate::common::CodegenCx;
use crate::coverageinfo;

use log::debug;
use rustc_codegen_ssa::coverageinfo::map::*;
use rustc_codegen_ssa::traits::{BaseTypeMethods, ConstMethods, MiscMethods};
use rustc_data_structures::fx::FxHashMap;
use rustc_llvm::RustString;
use rustc_middle::ty::Instance;
use rustc_middle::{bug, mir};

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;

// FIXME(richkadel): Complete all variations of generating and exporting the coverage map to LLVM.
// The current implementation is an initial foundation with basic capabilities (Counters, but not
// CounterExpressions, etc.).

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
    let mut coverage_writer = CoverageMappingWriter::new(cx);

    let function_coverage_map = cx.coverage_context().take_function_coverage_map();

    // Encode coverage mappings and generate function records
    let mut function_records = Vec::<&'ll llvm::Value>::new();
    let coverage_mappings_buffer = llvm::build_byte_buffer(|coverage_mappings_buffer| {
        for (instance, function_coverage) in function_coverage_map.into_iter() {
            if let Some(function_record) = coverage_writer.write_function_mappings_and_record(
                instance,
                function_coverage,
                coverage_mappings_buffer,
            ) {
                function_records.push(function_record);
            }
        }
    });

    // Encode all filenames covered in this module, ordered by `file_id`
    let filenames_buffer = llvm::build_byte_buffer(|filenames_buffer| {
        coverageinfo::write_filenames_section_to_buffer(
            &coverage_writer.filenames,
            filenames_buffer,
        );
    });

    if coverage_mappings_buffer.len() > 0 {
        // Generate the LLVM IR representation of the coverage map and store it in a well-known
        // global constant.
        coverage_writer.write_coverage_map(
            function_records,
            filenames_buffer,
            coverage_mappings_buffer,
        );
    }
}

struct CoverageMappingWriter<'a, 'll, 'tcx> {
    cx: &'a CodegenCx<'ll, 'tcx>,
    filenames: Vec<CString>,
    filename_to_index: FxHashMap<CString, u32>,
}

impl<'a, 'll, 'tcx> CoverageMappingWriter<'a, 'll, 'tcx> {
    fn new(cx: &'a CodegenCx<'ll, 'tcx>) -> Self {
        Self { cx, filenames: Vec::new(), filename_to_index: FxHashMap::<CString, u32>::default() }
    }

    /// For the given function, get the coverage region data, stream it to the given buffer, and
    /// then generate and return a new function record.
    fn write_function_mappings_and_record(
        &mut self,
        instance: Instance<'tcx>,
        mut function_coverage: FunctionCoverage,
        coverage_mappings_buffer: &RustString,
    ) -> Option<&'ll llvm::Value> {
        let cx = self.cx;
        let coverageinfo: &mir::CoverageInfo = cx.tcx.coverageinfo(instance.def_id());
        debug!(
            "Generate coverage map for: {:?}, num_counters: {}, num_expressions: {}",
            instance, coverageinfo.num_counters, coverageinfo.num_expressions
        );
        debug_assert!(coverageinfo.num_counters > 0);

        let regions_in_file_order = function_coverage.regions_in_file_order(cx.sess().source_map());
        if regions_in_file_order.len() == 0 {
            return None;
        }

        // Stream the coverage mapping regions for the function (`instance`) to the buffer, and
        // compute the data byte size used.
        let old_len = coverage_mappings_buffer.len();
        self.regions_to_mappings(regions_in_file_order, coverage_mappings_buffer);
        let mapping_data_size = coverage_mappings_buffer.len() - old_len;
        debug_assert!(mapping_data_size > 0);

        let mangled_function_name = cx.tcx.symbol_name(instance).to_string();
        let name_ref = coverageinfo::compute_hash(&mangled_function_name);
        let function_source_hash = function_coverage.source_hash();

        // Generate and return the function record
        let name_ref_val = cx.const_u64(name_ref);
        let mapping_data_size_val = cx.const_u32(mapping_data_size as u32);
        let func_hash_val = cx.const_u64(function_source_hash);
        Some(cx.const_struct(
            &[name_ref_val, mapping_data_size_val, func_hash_val],
            /*packed=*/ true,
        ))
    }

    /// For each coverage region, extract its coverage data from the earlier coverage analysis.
    /// Use LLVM APIs to convert the data into buffered bytes compliant with the LLVM Coverage
    /// Mapping format.
    fn regions_to_mappings(
        &mut self,
        regions_in_file_order: BTreeMap<PathBuf, BTreeMap<CoverageLoc, (usize, CoverageKind)>>,
        coverage_mappings_buffer: &RustString,
    ) {
        let mut virtual_file_mapping = Vec::new();
        let mut mapping_regions = coverageinfo::SmallVectorCounterMappingRegion::new();
        let mut expressions = coverageinfo::SmallVectorCounterExpression::new();

        for (file_id, (file_path, file_coverage_regions)) in
            regions_in_file_order.into_iter().enumerate()
        {
            let file_id = file_id as u32;
            let filename = CString::new(file_path.to_string_lossy().to_string())
                .expect("null error converting filename to C string");
            debug!("  file_id: {} = '{:?}'", file_id, filename);
            let filenames_index = match self.filename_to_index.get(&filename) {
                Some(index) => *index,
                None => {
                    let index = self.filenames.len() as u32;
                    self.filenames.push(filename.clone());
                    self.filename_to_index.insert(filename, index);
                    index
                }
            };
            virtual_file_mapping.push(filenames_index);

            let mut mapping_indexes = vec![0 as u32; file_coverage_regions.len()];
            for (mapping_index, (region_id, _)) in file_coverage_regions.values().enumerate() {
                mapping_indexes[*region_id] = mapping_index as u32;
            }

            for (region_loc, (region_id, region_kind)) in file_coverage_regions.into_iter() {
                let mapping_index = mapping_indexes[region_id];
                match region_kind {
                    CoverageKind::Counter => {
                        debug!(
                            "  Counter {}, file_id: {}, region_loc: {}",
                            mapping_index, file_id, region_loc
                        );
                        mapping_regions.push_from(
                            mapping_index,
                            file_id,
                            region_loc.start_line,
                            region_loc.start_col,
                            region_loc.end_line,
                            region_loc.end_col,
                        );
                    }
                    CoverageKind::CounterExpression(lhs, op, rhs) => {
                        debug!(
                            "  CounterExpression {} = {} {:?} {}, file_id: {}, region_loc: {:?}",
                            mapping_index, lhs, op, rhs, file_id, region_loc,
                        );
                        mapping_regions.push_from(
                            mapping_index,
                            file_id,
                            region_loc.start_line,
                            region_loc.start_col,
                            region_loc.end_line,
                            region_loc.end_col,
                        );
                        expressions.push_from(op, lhs, rhs);
                    }
                    CoverageKind::Unreachable => {
                        debug!(
                            "  Unreachable region, file_id: {}, region_loc: {:?}",
                            file_id, region_loc,
                        );
                        bug!("Unreachable region not expected and not yet handled!")
                        // FIXME(richkadel): implement and call
                        //   mapping_regions.push_from(...) for unreachable regions
                    }
                }
            }
        }

        // Encode and append the current function's coverage mapping data
        coverageinfo::write_mapping_to_buffer(
            virtual_file_mapping,
            expressions,
            mapping_regions,
            coverage_mappings_buffer,
        );
    }

    fn write_coverage_map(
        self,
        function_records: Vec<&'ll llvm::Value>,
        filenames_buffer: Vec<u8>,
        mut coverage_mappings_buffer: Vec<u8>,
    ) {
        let cx = self.cx;

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
