//! For each function that was instrumented for coverage, we need to embed its
//! corresponding coverage mapping metadata inside the `__llvm_covfun`[^win]
//! linker section of the final binary.
//!
//! [^win]: On Windows the section name is `.lcovfun`.

use std::ffi::CString;

use rustc_abi::Align;
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods as _, ConstCodegenMethods, StaticCodegenMethods,
};
use rustc_middle::mir::coverage::{
    BasicCoverageBlock, CovTerm, CoverageIdsInfo, Expression, FunctionCoverageInfo, Mapping,
    MappingKind, Op,
};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::Span;
use rustc_target::spec::HasTargetSpec;
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::mapgen::{GlobalFileTable, VirtualFileMapping, spans};
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
    global_file_table: &mut GlobalFileTable,
    instance: Instance<'tcx>,
    is_used: bool,
) -> Option<CovfunRecord<'tcx>> {
    let fn_cov_info = tcx.instance_mir(instance.def).function_coverage_info.as_deref()?;
    let ids_info = tcx.coverage_ids_info(instance.def)?;

    let expressions = prepare_expressions(ids_info);

    let mut covfun = CovfunRecord {
        mangled_function_name: tcx.symbol_name(instance).name,
        source_hash: if is_used { fn_cov_info.function_source_hash } else { 0 },
        is_used,
        virtual_file_mapping: VirtualFileMapping::default(),
        expressions,
        regions: ffi::Regions::default(),
    };

    fill_region_tables(tcx, global_file_table, fn_cov_info, ids_info, &mut covfun);

    if covfun.regions.has_no_regions() {
        debug!(?covfun, "function has no mappings to embed; skipping");
        return None;
    }

    Some(covfun)
}

/// Convert the function's coverage-counter expressions into a form suitable for FFI.
fn prepare_expressions(ids_info: &CoverageIdsInfo) -> Vec<ffi::CounterExpression> {
    let counter_for_term = ffi::Counter::from_term;

    // We know that LLVM will optimize out any unused expressions before
    // producing the final coverage map, so there's no need to do the same
    // thing on the Rust side unless we're confident we can do much better.
    // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)
    ids_info
        .expressions
        .iter()
        .map(move |&Expression { lhs, op, rhs }| ffi::CounterExpression {
            lhs: counter_for_term(lhs),
            kind: match op {
                Op::Add => ffi::ExprKind::Add,
                Op::Subtract => ffi::ExprKind::Subtract,
            },
            rhs: counter_for_term(rhs),
        })
        .collect::<Vec<_>>()
}

/// Populates the mapping region tables in the current function's covfun record.
fn fill_region_tables<'tcx>(
    tcx: TyCtxt<'tcx>,
    global_file_table: &mut GlobalFileTable,
    fn_cov_info: &'tcx FunctionCoverageInfo,
    ids_info: &'tcx CoverageIdsInfo,
    covfun: &mut CovfunRecord<'tcx>,
) {
    // Currently a function's mappings must all be in the same file, so use the
    // first mapping's span to determine the file.
    let source_map = tcx.sess.source_map();
    let Some(first_span) = (try { fn_cov_info.mappings.first()?.span }) else {
        debug_assert!(false, "function has no mappings: {:?}", covfun.mangled_function_name);
        return;
    };
    let source_file = source_map.lookup_source_file(first_span.lo());

    // Look up the global file ID for that file.
    let global_file_id = global_file_table.global_file_id_for_file(&source_file);

    // Associate that global file ID with a local file ID for this function.
    let local_file_id = covfun.virtual_file_mapping.local_id_for_global(global_file_id);

    let ffi::Regions { code_regions, branch_regions, mcdc_branch_regions, mcdc_decision_regions } =
        &mut covfun.regions;

    let make_cov_span =
        |span: Span| spans::make_coverage_span(local_file_id, source_map, &source_file, span);
    let discard_all = tcx.sess.coverage_discard_all_spans_in_codegen();

    // For each counter/region pair in this function+file, convert it to a
    // form suitable for FFI.
    for &Mapping { ref kind, span } in &fn_cov_info.mappings {
        // If this function is unused, replace all counters with zero.
        let counter_for_bcb = |bcb: BasicCoverageBlock| -> ffi::Counter {
            let term = if covfun.is_used {
                ids_info.term_for_bcb[bcb].expect("every BCB in a mapping was given a term")
            } else {
                CovTerm::Zero
            };
            ffi::Counter::from_term(term)
        };

        // Convert the `Span` into coordinates that we can pass to LLVM, or
        // discard the span if conversion fails. In rare, cases _all_ of a
        // function's spans are discarded, and the rest of coverage codegen
        // needs to handle that gracefully to avoid a repeat of #133606.
        // We don't have a good test case for triggering that organically, so
        // instead we set `-Zcoverage-options=discard-all-spans-in-codegen`
        // to force it to occur.
        let Some(cov_span) = make_cov_span(span) else { continue };
        if discard_all {
            continue;
        }

        match *kind {
            MappingKind::Code { bcb } => {
                code_regions.push(ffi::CodeRegion { cov_span, counter: counter_for_bcb(bcb) });
            }
            MappingKind::Branch { true_bcb, false_bcb } => {
                branch_regions.push(ffi::BranchRegion {
                    cov_span,
                    true_counter: counter_for_bcb(true_bcb),
                    false_counter: counter_for_bcb(false_bcb),
                });
            }
            MappingKind::MCDCBranch { true_bcb, false_bcb, mcdc_params } => {
                mcdc_branch_regions.push(ffi::MCDCBranchRegion {
                    cov_span,
                    true_counter: counter_for_bcb(true_bcb),
                    false_counter: counter_for_bcb(false_bcb),
                    mcdc_branch_params: ffi::mcdc::BranchParameters::from(mcdc_params),
                });
            }
            MappingKind::MCDCDecision(mcdc_decision_params) => {
                mcdc_decision_regions.push(ffi::MCDCDecisionRegion {
                    cov_span,
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
