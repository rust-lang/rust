//! For each function that was instrumented for coverage, we need to embed its
//! corresponding coverage mapping metadata inside the `__llvm_covfun`[^win]
//! linker section of the final binary.
//!
//! [^win]: On Windows the section name is `.lcovfun`.

use std::ffi::CString;

use rustc_abi::Align;
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_middle::bug;
use rustc_middle::mir::coverage::MappingKind;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_target::spec::HasTargetSpec;
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::map_data::FunctionCoverage;
use crate::coverageinfo::mapgen::{GlobalFileTable, VirtualFileMapping, span_file_name};
use crate::coverageinfo::{ffi, llvm_cov};
use crate::llvm;

/// Intermediate coverage metadata for a single function, used to help build
/// the final record that will be embedded in the `__llvm_covfun` section.
#[derive(Debug)]
pub(crate) struct CovfunRecord<'tcx> {
    mangled_function_name: &'tcx str,
    source_hash: u64,
    is_used: bool,

    virtual_file_mapping: VirtualFileMapping,
    expressions: Vec<ffi::CounterExpression>,
    regions: ffi::Regions,
}

impl<'tcx> CovfunRecord<'tcx> {
    /// FIXME(Zalathar): Make this the responsibility of the code that determines
    /// which functions are unused.
    pub(crate) fn mangled_function_name_if_unused(&self) -> Option<&'tcx str> {
        (!self.is_used).then_some(self.mangled_function_name)
    }
}

pub(crate) fn prepare_covfun_record<'tcx>(
    tcx: TyCtxt<'tcx>,
    global_file_table: &GlobalFileTable,
    instance: Instance<'tcx>,
    function_coverage: &FunctionCoverage<'tcx>,
) -> Option<CovfunRecord<'tcx>> {
    let mut covfun = CovfunRecord {
        mangled_function_name: tcx.symbol_name(instance).name,
        source_hash: function_coverage.source_hash(),
        is_used: function_coverage.is_used(),
        virtual_file_mapping: VirtualFileMapping::default(),
        expressions: function_coverage.counter_expressions().collect::<Vec<_>>(),
        regions: ffi::Regions::default(),
    };

    fill_region_tables(tcx, global_file_table, function_coverage, &mut covfun);

    if covfun.regions.has_no_regions() {
        if covfun.is_used {
            bug!("a used function should have had coverage mapping data but did not: {covfun:?}");
        } else {
            debug!(?covfun, "unused function had no coverage mapping data");
            return None;
        }
    }

    Some(covfun)
}

/// Populates the mapping region tables in the current function's covfun record.
fn fill_region_tables<'tcx>(
    tcx: TyCtxt<'tcx>,
    global_file_table: &GlobalFileTable,
    function_coverage: &FunctionCoverage<'tcx>,
    covfun: &mut CovfunRecord<'tcx>,
) {
    let counter_regions = function_coverage.counter_regions();
    if counter_regions.is_empty() {
        return;
    }

    // Currently a function's mappings must all be in the same file as its body span.
    let file_name = span_file_name(tcx, function_coverage.function_coverage_info.body_span);

    // Look up the global file ID for that filename.
    let global_file_id = global_file_table.global_file_id_for_file_name(file_name);

    // Associate that global file ID with a local file ID for this function.
    let local_file_id = covfun.virtual_file_mapping.local_id_for_global(global_file_id);
    debug!("  file id: {local_file_id:?} => {global_file_id:?} = '{file_name:?}'");

    let ffi::Regions { code_regions, branch_regions, mcdc_branch_regions, mcdc_decision_regions } =
        &mut covfun.regions;

    // For each counter/region pair in this function+file, convert it to a
    // form suitable for FFI.
    for (mapping_kind, region) in counter_regions {
        debug!("Adding counter {mapping_kind:?} to map for {region:?}");
        let span = ffi::CoverageSpan::from_source_region(local_file_id, region);
        match mapping_kind {
            MappingKind::Code(term) => {
                code_regions.push(ffi::CodeRegion { span, counter: ffi::Counter::from_term(term) });
            }
            MappingKind::Branch { true_term, false_term } => {
                branch_regions.push(ffi::BranchRegion {
                    span,
                    true_counter: ffi::Counter::from_term(true_term),
                    false_counter: ffi::Counter::from_term(false_term),
                });
            }
            MappingKind::MCDCBranch { true_term, false_term, mcdc_params } => {
                mcdc_branch_regions.push(ffi::MCDCBranchRegion {
                    span,
                    true_counter: ffi::Counter::from_term(true_term),
                    false_counter: ffi::Counter::from_term(false_term),
                    mcdc_branch_params: ffi::mcdc::BranchParameters::from(mcdc_params),
                });
            }
            MappingKind::MCDCDecision(mcdc_decision_params) => {
                mcdc_decision_regions.push(ffi::MCDCDecisionRegion {
                    span,
                    mcdc_decision_params: ffi::mcdc::DecisionParameters::from(mcdc_decision_params),
                });
            }
        }
    }
}

/// Generates the contents of the covfun record for this function, which
/// contains the function's coverage mapping data. The record is then stored
/// as a global variable in the `__llvm_covfun` section.
pub(crate) fn generate_covfun_record<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    filenames_hash: u64,
    covfun: &CovfunRecord<'tcx>,
) {
    let &CovfunRecord {
        mangled_function_name,
        source_hash,
        is_used,
        ref virtual_file_mapping,
        ref expressions,
        ref regions,
    } = covfun;

    // Encode the function's coverage mappings into a buffer.
    let coverage_mapping_buffer = llvm_cov::write_function_mappings_to_buffer(
        &virtual_file_mapping.to_vec(),
        expressions,
        regions,
    );

    // A covfun record consists of four target-endian integers, followed by the
    // encoded mapping data in bytes. Note that the length field is 32 bits.
    // <https://llvm.org/docs/CoverageMappingFormat.html#llvm-ir-representation>
    // See also `src/llvm-project/clang/lib/CodeGen/CoverageMappingGen.cpp` and
    // `COVMAP_V3` in `src/llvm-project/llvm/include/llvm/ProfileData/InstrProfData.inc`.
    let func_name_hash = llvm_cov::hash_bytes(mangled_function_name.as_bytes());
    let covfun_record = cx.const_struct(
        &[
            cx.const_u64(func_name_hash),
            cx.const_u32(coverage_mapping_buffer.len() as u32),
            cx.const_u64(source_hash),
            cx.const_u64(filenames_hash),
            cx.const_bytes(&coverage_mapping_buffer),
        ],
        // This struct needs to be packed, so that the 32-bit length field
        // doesn't have unexpected padding.
        true,
    );

    // Choose a variable name to hold this function's covfun data.
    // Functions that are used have a suffix ("u") to distinguish them from
    // unused copies of the same function (from different CGUs), so that if a
    // linker sees both it won't discard the used copy's data.
    let u = if is_used { "u" } else { "" };
    let covfun_var_name = CString::new(format!("__covrec_{func_name_hash:X}{u}")).unwrap();
    debug!("function record var name: {covfun_var_name:?}");

    let covfun_global = llvm::add_global(cx.llmod, cx.val_ty(covfun_record), &covfun_var_name);
    llvm::set_initializer(covfun_global, covfun_record);
    llvm::set_global_constant(covfun_global, true);
    llvm::set_linkage(covfun_global, llvm::Linkage::LinkOnceODRLinkage);
    llvm::set_visibility(covfun_global, llvm::Visibility::Hidden);
    llvm::set_section(covfun_global, cx.covfun_section_name());
    // LLVM's coverage mapping format specifies 8-byte alignment for items in this section.
    // <https://llvm.org/docs/CoverageMappingFormat.html>
    llvm::set_alignment(covfun_global, Align::EIGHT);
    if cx.target_spec().supports_comdat() {
        llvm::set_comdat(cx.llmod, covfun_global, &covfun_var_name);
    }

    cx.add_used_global(covfun_global);
}
