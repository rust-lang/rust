use crate::common::CodegenCx;
use crate::coverageinfo;
use crate::coverageinfo::map_data::{Counter, CounterExpression};
use crate::llvm;

use llvm::coverageinfo::CounterMappingRegion;
use rustc_codegen_ssa::traits::ConstMethods;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_llvm::RustString;
use rustc_middle::bug;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::coverage::CodeRegion;
use rustc_middle::ty::TyCtxt;

use std::ffi::CString;

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

    let mut mapgen = CoverageMapGenerator::new(tcx);

    // Encode coverage mappings and generate function records
    let mut function_data = Vec::new();
    for (instance, function_coverage) in function_coverage_map {
        debug!("Generate function coverage for {}, {:?}", cx.codegen_unit.name(), instance);
        let mangled_function_name = tcx.symbol_name(instance).name;
        let source_hash = function_coverage.source_hash();
        let is_used = function_coverage.is_used();
        let (expressions, counter_regions) =
            function_coverage.get_expressions_and_counter_regions();

        let coverage_mapping_buffer = llvm::build_byte_buffer(|coverage_mapping_buffer| {
            mapgen.write_coverage_mapping(expressions, counter_regions, coverage_mapping_buffer);
        });

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
    let filenames_buffer = llvm::build_byte_buffer(|filenames_buffer| {
        coverageinfo::write_filenames_section_to_buffer(&mapgen.filenames, filenames_buffer);
    });

    let filenames_size = filenames_buffer.len();
    let filenames_val = cx.const_bytes(&filenames_buffer);
    let filenames_ref = coverageinfo::hash_bytes(&filenames_buffer);

    // Generate the LLVM IR representation of the coverage map and store it in a well-known global
    let cov_data_val = mapgen.generate_coverage_map(cx, version, filenames_size, filenames_val);

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

struct CoverageMapGenerator {
    filenames: FxIndexSet<CString>,
}

impl CoverageMapGenerator {
    fn new(tcx: TyCtxt<'_>) -> Self {
        let mut filenames = FxIndexSet::default();
        // LLVM Coverage Mapping Format version 6 (zero-based encoded as 5)
        // requires setting the first filename to the compilation directory.
        // Since rustc generates coverage maps with relative paths, the
        // compilation directory can be combined with the relative paths
        // to get absolute paths, if needed.
        let working_dir =
            tcx.sess.opts.working_dir.remapped_path_if_available().to_string_lossy().to_string();
        let c_filename =
            CString::new(working_dir).expect("null error converting filename to C string");
        filenames.insert(c_filename);
        Self { filenames }
    }

    /// Using the `expressions` and `counter_regions` collected for the current function, generate
    /// the `mapping_regions` and `virtual_file_mapping`, and capture any new filenames. Then use
    /// LLVM APIs to encode the `virtual_file_mapping`, `expressions`, and `mapping_regions` into
    /// the given `coverage_mapping` byte buffer, compliant with the LLVM Coverage Mapping format.
    fn write_coverage_mapping<'a>(
        &mut self,
        expressions: Vec<CounterExpression>,
        counter_regions: impl Iterator<Item = (Counter, &'a CodeRegion)>,
        coverage_mapping_buffer: &RustString,
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
            let same_file = current_file_name.is_some_and(|p| p == file_name);
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
            debug!("Adding counter {:?} to map for {:?}", counter, region);
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
            coverage_mapping_buffer,
        );
    }

    /// Construct coverage map header and the array of function records, and combine them into the
    /// coverage map. Save the coverage map data into the LLVM IR as a static global using a
    /// specific, well-known section and name.
    fn generate_coverage_map<'ll>(
        self,
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
/// `DefId`s (`tcx` query `mir_keys`) minus the codegenned `DefId`s (`tcx` query
/// `codegened_and_inlined_items`).
///
/// These unused functions are then codegen'd in one of the CGUs which is marked as the
/// "code coverage dead code cgu" during the partitioning process. This prevents us from generating
/// code regions for the same function more than once which can lead to linker errors regarding
/// duplicate symbols.
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
                DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Generator
            ) {
                return None;
            }
            if ignore_unused_generics && tcx.generics_of(def_id).requires_monomorphization(tcx) {
                return None;
            }
            Some(local_def_id.to_def_id())
        })
        .collect();

    let codegenned_def_ids = tcx.codegened_and_inlined_items(());

    for non_codegenned_def_id in
        eligible_def_ids.into_iter().filter(|id| !codegenned_def_ids.contains(id))
    {
        let codegen_fn_attrs = tcx.codegen_fn_attrs(non_codegenned_def_id);

        // If a function is marked `#[no_coverage]`, then skip generating a
        // dead code stub for it.
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_COVERAGE) {
            debug!("skipping unused fn marked #[no_coverage]: {:?}", non_codegenned_def_id);
            continue;
        }

        debug!("generating unused fn: {:?}", non_codegenned_def_id);
        cx.define_unused_fn(non_codegenned_def_id);
    }
}
