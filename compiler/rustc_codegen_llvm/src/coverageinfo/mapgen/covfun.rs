//! For each function that was instrumented for coverage, we need to embed its
//! corresponding coverage mapping metadata inside the `__llvm_covfun`[^win]
//! linker section of the final binary.
//!
//! [^win]: On Windows the section name is `.lcovfun`.

use std::ffi::CString;
use std::sync::Arc;

use rustc_abi::Align;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods as _, ConstCodegenMethods};
use rustc_middle::mir::coverage::{
    BasicCoverageBlock, CounterId, CovTerm, CoverageIdsInfo, Expression, ExpressionId,
    FunctionCoverageInfo, Mapping, MappingKind, Op,
};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::{SourceFile, Span};
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
    /// Not used directly, but helpful in debug messages.
    _instance: Instance<'tcx>,

    mangled_function_name: &'tcx str,
    source_hash: u64,
    is_used: bool,

    virtual_file_mapping: VirtualFileMapping,
    expressions: Vec<ffi::CounterExpression>,
    regions: llvm_cov::Regions,
}

impl<'tcx> CovfunRecord<'tcx> {
    /// Iterator that yields all source files referred to by this function's
    /// coverage mappings. Used to build the global file table for the CGU.
    pub(crate) fn all_source_files(&self) -> impl Iterator<Item = &SourceFile> {
        self.virtual_file_mapping.local_file_table.iter().map(Arc::as_ref)
    }
}

pub(crate) fn prepare_covfun_record<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    is_used: bool,
) -> Option<CovfunRecord<'tcx>> {
    let fn_cov_info = tcx.instance_mir(instance.def).function_coverage_info.as_deref()?;
    let ids_info = tcx.coverage_ids_info(instance.def)?;

    let expressions = prepare_expressions(ids_info);

    let mut covfun = CovfunRecord {
        _instance: instance,
        mangled_function_name: tcx.symbol_name(instance).name,
        source_hash: if is_used { fn_cov_info.function_source_hash } else { 0 },
        is_used,
        virtual_file_mapping: VirtualFileMapping::default(),
        expressions,
        regions: llvm_cov::Regions::default(),
    };

    fill_region_tables(tcx, fn_cov_info, ids_info, &mut covfun);

    if covfun.regions.has_no_regions() {
        debug!(?covfun, "function has no mappings to embed; skipping");
        return None;
    }

    Some(covfun)
}

pub(crate) fn counter_for_term(term: CovTerm) -> ffi::Counter {
    use ffi::Counter;
    match term {
        CovTerm::Zero => Counter::ZERO,
        CovTerm::Counter(id) => {
            Counter { kind: ffi::CounterKind::CounterValueReference, id: CounterId::as_u32(id) }
        }
        CovTerm::Expression(id) => {
            Counter { kind: ffi::CounterKind::Expression, id: ExpressionId::as_u32(id) }
        }
    }
}

/// Convert the function's coverage-counter expressions into a form suitable for FFI.
fn prepare_expressions(ids_info: &CoverageIdsInfo) -> Vec<ffi::CounterExpression> {
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
    fn_cov_info: &'tcx FunctionCoverageInfo,
    ids_info: &'tcx CoverageIdsInfo,
    covfun: &mut CovfunRecord<'tcx>,
) {
    // If this function is unused, replace all counters with zero.
    let counter_for_bcb = |bcb: BasicCoverageBlock| -> ffi::Counter {
        let term = if covfun.is_used {
            ids_info.term_for_bcb[bcb].expect("every BCB in a mapping was given a term")
        } else {
            CovTerm::Zero
        };
        counter_for_term(term)
    };

    // Currently a function's mappings must all be in the same file, so use the
    // first mapping's span to determine the file.
    let source_map = tcx.sess.source_map();
    let Some(first_span) = (try { fn_cov_info.mappings.first()?.span }) else {
        debug_assert!(false, "function has no mappings: {covfun:?}");
        return;
    };
    let source_file = source_map.lookup_source_file(first_span.lo());

    let local_file_id = covfun.virtual_file_mapping.push_file(&source_file);

    // In rare cases, _all_ of a function's spans are discarded, and coverage
    // codegen needs to handle that gracefully to avoid #133606.
    // It's hard for tests to trigger this organically, so instead we set
    // `-Zcoverage-options=discard-all-spans-in-codegen` to force it to occur.
    let discard_all = tcx.sess.coverage_options().discard_all_spans_in_codegen;
    let make_coords = |span: Span| {
        if discard_all { None } else { spans::make_coords(source_map, &source_file, span) }
    };

    let llvm_cov::Regions {
        code_regions,
        expansion_regions: _, // FIXME(Zalathar): Fill out support for expansion regions
        branch_regions,
    } = &mut covfun.regions;

    // For each counter/region pair in this function+file, convert it to a
    // form suitable for FFI.
    for &Mapping { ref kind, span } in &fn_cov_info.mappings {
        let Some(coords) = make_coords(span) else { continue };
        let cov_span = coords.make_coverage_span(local_file_id);

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
        }
    }
}

/// Generates the contents of the covfun record for this function, which
/// contains the function's coverage mapping data. The record is then stored
/// as a global variable in the `__llvm_covfun` section.
pub(crate) fn generate_covfun_record<'tcx>(
    cx: &mut CodegenCx<'_, 'tcx>,
    global_file_table: &GlobalFileTable,
    covfun: &CovfunRecord<'tcx>,
) {
    let &CovfunRecord {
        _instance,
        mangled_function_name,
        source_hash,
        is_used,
        ref virtual_file_mapping,
        ref expressions,
        ref regions,
    } = covfun;

    let Some(local_file_table) = virtual_file_mapping.resolve_all(global_file_table) else {
        debug_assert!(
            false,
            "all local files should be present in the global file table: \
                global_file_table = {global_file_table:?}, \
                virtual_file_mapping = {virtual_file_mapping:?}"
        );
        return;
    };

    // Encode the function's coverage mappings into a buffer.
    let coverage_mapping_buffer =
        llvm_cov::write_function_mappings_to_buffer(&local_file_table, expressions, regions);

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
            cx.const_u64(global_file_table.filenames_hash),
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
