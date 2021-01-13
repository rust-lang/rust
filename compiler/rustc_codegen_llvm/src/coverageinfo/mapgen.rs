use crate::common::CodegenCx;
use crate::coverageinfo;
use crate::llvm;

use llvm::coverageinfo::CounterMappingRegion;
use rustc_codegen_ssa::coverageinfo::map::{Counter, CounterExpression, FunctionCoverage};
use rustc_codegen_ssa::traits::ConstMethods;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_hir::def_id::{DefId, DefIdSet, LOCAL_CRATE};
use rustc_llvm::RustString;
use rustc_middle::mir::coverage::CodeRegion;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::Symbol;

use std::ffi::CString;

use tracing::debug;

/// Generates and exports the Coverage Map.
///
/// This Coverage Map complies with Coverage Mapping Format version 4 (zero-based encoded as 3),
/// as defined at [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/rustc/11.0-2020-10-12/llvm/docs/CoverageMappingFormat.rst#llvm-code-coverage-mapping-format)
/// and published in Rust's current (November 2020) fork of LLVM. This version is supported by the
/// LLVM coverage tools (`llvm-profdata` and `llvm-cov`) bundled with Rust's fork of LLVM.
///
/// Consequently, Rust's bundled version of Clang also generates Coverage Maps compliant with
/// version 3. Clang's implementation of Coverage Map generation was referenced when implementing
/// this Rust version, and though the format documentation is very explicit and detailed, some
/// undocumented details in Clang's implementation (that may or may not be important) were also
/// replicated for Rust's Coverage Map.
pub fn finalize<'ll, 'tcx>(cx: &CodegenCx<'ll, 'tcx>) {
    let tcx = cx.tcx;
    // Ensure LLVM supports Coverage Map Version 4 (encoded as a zero-based value: 3).
    // If not, the LLVM Version must be less than 11.
    let version = coverageinfo::mapping_version();
    if version != 3 {
        tcx.sess.fatal("rustc option `-Z instrument-coverage` requires LLVM 11 or higher.");
    }

    debug!("Generating coverage map for CodegenUnit: `{}`", cx.codegen_unit.name());

    let mut function_coverage_map = match cx.coverage_context() {
        Some(ctx) => ctx.take_function_coverage_map(),
        None => return,
    };
    if function_coverage_map.is_empty() {
        // This module has no functions with coverage instrumentation
        return;
    }

    add_unreachable_coverage(tcx, &mut function_coverage_map);

    let mut mapgen = CoverageMapGenerator::new();

    // Encode coverage mappings and generate function records
    let mut function_data = Vec::new();
    for (instance, function_coverage) in function_coverage_map {
        debug!("Generate function coverage for {}, {:?}", cx.codegen_unit.name(), instance);
        let mangled_function_name = tcx.symbol_name(instance).to_string();
        let function_source_hash = function_coverage.source_hash();
        let (expressions, counter_regions) =
            function_coverage.get_expressions_and_counter_regions();

        let coverage_mapping_buffer = llvm::build_byte_buffer(|coverage_mapping_buffer| {
            mapgen.write_coverage_mapping(expressions, counter_regions, coverage_mapping_buffer);
        });
        debug_assert!(
            coverage_mapping_buffer.len() > 0,
            "Every `FunctionCoverage` should have at least one counter"
        );

        function_data.push((mangled_function_name, function_source_hash, coverage_mapping_buffer));
    }

    // Encode all filenames referenced by counters/expressions in this module
    let filenames_buffer = llvm::build_byte_buffer(|filenames_buffer| {
        coverageinfo::write_filenames_section_to_buffer(&mapgen.filenames, filenames_buffer);
    });

    let filenames_size = filenames_buffer.len();
    let filenames_val = cx.const_bytes(&filenames_buffer[..]);
    let filenames_ref = coverageinfo::hash_bytes(filenames_buffer);

    // Generate the LLVM IR representation of the coverage map and store it in a well-known global
    let cov_data_val = mapgen.generate_coverage_map(cx, version, filenames_size, filenames_val);

    for (mangled_function_name, function_source_hash, coverage_mapping_buffer) in function_data {
        save_function_record(
            cx,
            mangled_function_name,
            function_source_hash,
            filenames_ref,
            coverage_mapping_buffer,
        );
    }

    // Save the coverage data value to LLVM IR
    coverageinfo::save_cov_data_to_mod(cx, cov_data_val);
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
    /// the given `coverage_mapping` byte buffer, compliant with the LLVM Coverage Mapping format.
    fn write_coverage_mapping(
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
    fn generate_coverage_map(
        self,
        cx: &CodegenCx<'ll, 'tcx>,
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
    cx: &CodegenCx<'ll, 'tcx>,
    mangled_function_name: String,
    function_source_hash: u64,
    filenames_ref: u64,
    coverage_mapping_buffer: Vec<u8>,
) {
    // Concatenate the encoded coverage mappings
    let coverage_mapping_size = coverage_mapping_buffer.len();
    let coverage_mapping_val = cx.const_bytes(&coverage_mapping_buffer[..]);

    let func_name_hash = coverageinfo::hash_str(&mangled_function_name);
    let func_name_hash_val = cx.const_u64(func_name_hash);
    let coverage_mapping_size_val = cx.const_u32(coverage_mapping_size as u32);
    let func_hash_val = cx.const_u64(function_source_hash);
    let filenames_ref_val = cx.const_u64(filenames_ref);
    let func_record_val = cx.const_struct(
        &[
            func_name_hash_val,
            coverage_mapping_size_val,
            func_hash_val,
            filenames_ref_val,
            coverage_mapping_val,
        ],
        /*packed=*/ true,
    );

    // At the present time, the coverage map for Rust assumes every instrumented function `is_used`.
    // Note that Clang marks functions as "unused" in `CodeGenPGO::emitEmptyCounterMapping`. (See:
    // https://github.com/rust-lang/llvm-project/blob/de02a75e398415bad4df27b4547c25b896c8bf3b/clang%2Flib%2FCodeGen%2FCodeGenPGO.cpp#L877-L878
    // for example.)
    //
    // It's not yet clear if or how this may be applied to Rust in the future, but the `is_used`
    // argument is available and handled similarly.
    let is_used = true;
    coverageinfo::save_func_record_to_mod(cx, func_name_hash, func_record_val, is_used);
}

/// When finalizing the coverage map, `FunctionCoverage` only has the `CodeRegion`s and counters for
/// the functions that went through codegen; such as public functions and "used" functions
/// (functions referenced by other "used" or public items). Any other functions considered unused,
/// or "Unreachable" were still parsed and processed through the MIR stage.
///
/// We can find the unreachable functions by the set difference of all MIR `DefId`s (`tcx` query
/// `mir_keys`) minus the codegenned `DefId`s (`tcx` query `collect_and_partition_mono_items`).
///
/// *HOWEVER* the codegenned `DefId`s are partitioned across multiple `CodegenUnit`s (CGUs), and
/// this function is processing a `function_coverage_map` for the functions (`Instance`/`DefId`)
/// allocated to only one of those CGUs. We must NOT inject any "Unreachable" functions's
/// `CodeRegion`s more than once, so we have to pick which CGU's `function_coverage_map` to add
/// each "Unreachable" function to.
///
/// Some constraints:
///
/// 1. The file name of an "Unreachable" function must match the file name of the existing
///    codegenned (covered) function to which the unreachable code regions will be added.
/// 2. The function to which the unreachable code regions will be added must not be a genaric
///    function (must not have type parameters) because the coverage tools will get confused
///    if the codegenned function has more than one instantiation and additional `CodeRegion`s
///    attached to only one of those instantiations.
fn add_unreachable_coverage<'tcx>(
    tcx: TyCtxt<'tcx>,
    function_coverage_map: &mut FxHashMap<Instance<'tcx>, FunctionCoverage<'tcx>>,
) {
    // FIXME(#79622): Can this solution be simplified and/or improved? Are there other sources
    // of compiler state data that might help (or better sources that could be exposed, but
    // aren't yet)?

    // Note: If the crate *only* defines generic functions, there are no codegenerated non-generic
    // functions to add any unreachable code to. In this case, the unreachable code regions will
    // have no coverage, instead of having coverage with zero executions.
    //
    // This is probably still an improvement over Clang, which does not generate any coverage
    // for uninstantiated template functions.

    let has_non_generic_def_ids =
        function_coverage_map.keys().any(|instance| instance.def.attrs(tcx).len() == 0);

    if !has_non_generic_def_ids {
        // There are no non-generic functions to add unreachable `CodeRegion`s to
        return;
    }

    let all_def_ids: DefIdSet =
        tcx.mir_keys(LOCAL_CRATE).iter().map(|local_def_id| local_def_id.to_def_id()).collect();

    let (codegenned_def_ids, _) = tcx.collect_and_partition_mono_items(LOCAL_CRATE);

    let mut unreachable_def_ids_by_file: FxHashMap<Symbol, Vec<DefId>> = FxHashMap::default();
    for &non_codegenned_def_id in all_def_ids.difference(codegenned_def_ids) {
        // Make sure the non-codegenned (unreachable) function has a file_name
        if let Some(non_codegenned_file_name) = tcx.covered_file_name(non_codegenned_def_id) {
            let def_ids = unreachable_def_ids_by_file
                .entry(*non_codegenned_file_name)
                .or_insert_with(Vec::new);
            def_ids.push(non_codegenned_def_id);
        }
    }

    if unreachable_def_ids_by_file.is_empty() {
        // There are no unreachable functions with file names to add (in any CGU)
        return;
    }

    // Since there may be multiple `CodegenUnit`s, some codegenned_def_ids may be codegenned in a
    // different CGU, and will be added to the function_coverage_map for each CGU. Determine which
    // function_coverage_map has the responsibility for publishing unreachable coverage
    // based on file name:
    //
    // For each covered file name, sort ONLY the non-generic codegenned_def_ids, and if
    // covered_def_ids.contains(the first def_id) for a given file_name, add the unreachable code
    // region in this function_coverage_map. Otherwise, ignore it and assume another CGU's
    // function_coverage_map will be adding it (because it will be first for one, and only one,
    // of them).
    let mut sorted_codegenned_def_ids: Vec<DefId> =
        codegenned_def_ids.iter().map(|def_id| *def_id).collect();
    sorted_codegenned_def_ids.sort_unstable();

    let mut first_covered_def_id_by_file: FxHashMap<Symbol, DefId> = FxHashMap::default();
    for &def_id in sorted_codegenned_def_ids.iter() {
        // Only consider non-generic functions, to potentially add unreachable code regions
        if tcx.generics_of(def_id).count() == 0 {
            if let Some(covered_file_name) = tcx.covered_file_name(def_id) {
                // Only add files known to have unreachable functions
                if unreachable_def_ids_by_file.contains_key(covered_file_name) {
                    first_covered_def_id_by_file.entry(*covered_file_name).or_insert(def_id);
                }
            }
        }
    }

    // Get the set of def_ids with coverage regions, known by *this* CoverageContext.
    let cgu_covered_def_ids: DefIdSet =
        function_coverage_map.keys().map(|instance| instance.def.def_id()).collect();

    let mut cgu_covered_files: FxHashSet<Symbol> = first_covered_def_id_by_file
        .iter()
        .filter_map(
            |(&file_name, def_id)| {
                if cgu_covered_def_ids.contains(def_id) { Some(file_name) } else { None }
            },
        )
        .collect();

    // Find the first covered, non-generic function (instance) for each cgu_covered_file. Take the
    // unreachable code regions for that file, and add them to the function.
    //
    // There are three `for` loops here, but (a) the lists have already been reduced to the minimum
    // required values, the lists are further reduced (by `remove()` calls) when elements are no
    // longer needed, and there are several opportunities to branch out of loops early.
    for (instance, function_coverage) in function_coverage_map.iter_mut() {
        if instance.def.attrs(tcx).len() > 0 {
            continue;
        }
        // The covered function is not generic...
        let covered_def_id = instance.def.def_id();
        if let Some(covered_file_name) = tcx.covered_file_name(covered_def_id) {
            if !cgu_covered_files.remove(&covered_file_name) {
                continue;
            }
            // The covered function's file is one of the files with unreachable code regions, so
            // all of the unreachable code regions for this file will be added to this function.
            for def_id in
                unreachable_def_ids_by_file.remove(&covered_file_name).into_iter().flatten()
            {
                // Note, this loop adds an unreachable code regions for each MIR-derived region.
                // Alternatively, we could add a single code region for the maximum span of all
                // code regions here.
                //
                // Observed downsides of this approach are:
                //
                // 1. The coverage results will appear inconsistent compared with the same (or
                //    similar) code in a function that is reached.
                // 2. If the function is unreachable from one crate but reachable when compiling
                //    another referencing crate (such as a cross-crate reference to a
                //    generic function or inlined function), actual coverage regions overlaid
                //    on a single larger code span of `Zero` coverage can appear confusing or
                //    wrong. Chaning the unreachable coverage from a `code_region` to a
                //    `gap_region` can help, but still can look odd with `0` line counts for
                //    lines between executed (> 0) lines (such as for blank lines or comments).
                for &region in tcx.covered_code_regions(def_id) {
                    function_coverage.add_unreachable_region(region.clone());
                }
            }
            if cgu_covered_files.is_empty() {
                break;
            }
        }
    }
}
