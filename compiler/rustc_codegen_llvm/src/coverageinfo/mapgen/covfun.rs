//! For each function that was instrumented for coverage, we need to embed its
//! corresponding coverage mapping metadata inside the `__llvm_covfun`[^win]
//! linker section of the final binary.
//!
//! [^win]: On Windows the section name is `.lcovfun`.

use std::ffi::CString;
use std::iter;
use std::sync::Arc;

use rustc_abi::Align;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods as _, ConstCodegenMethods};
use rustc_index::IndexVec;
use rustc_middle::mir::coverage::{
    BasicCoverageBlock, CounterId, CovTerm, CoverageCodegenInfo, Expression, ExpressionId, Mapping,
    MappingKind, Op,
};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::{SourceFile, Span};
use rustc_target::spec::HasTargetSpec;
use tracing::debug;

use crate::common::CodegenCx;
use crate::coverageinfo::mapgen::{GlobalFileTable, LocalFileId, spans};
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

    expressions: Vec<ffi::CounterExpression>,
    mappings: ResolvedMappings,
}

impl<'tcx> CovfunRecord<'tcx> {
    /// Iterator that yields all source files referred to by this function's
    /// coverage mappings. Used to build the global file table for the CGU.
    pub(crate) fn all_source_files(&self) -> impl Iterator<Item = &SourceFile> {
        self.mappings.all_source_files()
    }
}

pub(crate) fn prepare_covfun_record<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    is_used: bool,
) -> Option<CovfunRecord<'tcx>> {
    let mir_info = tcx.instance_mir(instance.def).coverage_mir_info.as_deref()?;
    let cg_info = tcx.coverage_codegen_info(instance.def)?;

    let expressions = prepare_expressions(cg_info);
    let mappings = prepare_resolved_mappings(tcx, cg_info, is_used, &mir_info.mappings)?;

    let covfun = CovfunRecord {
        _instance: instance,
        mangled_function_name: tcx.symbol_name(instance).name,
        source_hash: if is_used { mir_info.function_source_hash } else { 0 },
        is_used,
        expressions,
        mappings,
    };

    Some(covfun)
}

fn counter_for_term(term: CovTerm) -> ffi::Counter {
    match term {
        CovTerm::Zero => ffi::Counter::ZERO,
        CovTerm::Counter(id) => ffi::Counter {
            kind: ffi::CounterKind::CounterValueReference,
            id: CounterId::as_u32(id),
        },
        CovTerm::Expression(id) => {
            ffi::Counter { kind: ffi::CounterKind::Expression, id: ExpressionId::as_u32(id) }
        }
    }
}

/// Convert the function's coverage-counter expressions into a form suitable for FFI.
fn prepare_expressions(cg_info: &CoverageCodegenInfo) -> Vec<ffi::CounterExpression> {
    // We know that LLVM will optimize out any unused expressions before
    // producing the final coverage map, so there's no need to do the same
    // thing on the Rust side unless we're confident we can do much better.
    // (See `CounterExpressionsMinimizer` in `CoverageMappingWriter.cpp`.)
    cg_info
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

/// Intermediate representation of coverage mappings, after all mapping spans
/// have been resolved to file coordinates (or discarded), but before producing
/// a final [`llvm_cov::Regions`].
///
/// Having a separate resolution step makes it easier to handle edge cases
/// where a function (or someday an expansion) manages to lose all of its spans,
/// without accidentally emitting invalid covfun records containing empty files.
#[derive(Debug)]
struct ResolvedMappings {
    /// Source file for all of the [`spans::Coords`] in these mappings.
    source_file: Arc<SourceFile>,

    code_mappings: Vec<CodeMapping>,
    branch_mappings: Vec<BranchMapping>,
}

impl ResolvedMappings {
    fn ensure_nonempty(self) -> Option<Self> {
        let ResolvedMappings { source_file: _, code_mappings, branch_mappings } = &self;
        if code_mappings.is_empty() && branch_mappings.is_empty() { None } else { Some(self) }
    }

    fn all_source_files(&self) -> impl Iterator<Item = &SourceFile> {
        // FIXME(Zalathar): When expansion regions are supported, this also needs to yield
        // any source files used by descendant expansions.
        let ResolvedMappings { source_file, code_mappings: _, branch_mappings: _ } = self;
        iter::once(source_file.as_ref())
    }
}

/// Resolved from [`MappingKind::Code`], and the precursor to [`ffi::CodeRegion`].
#[derive(Debug)]
struct CodeMapping {
    coords: spans::Coords,
    counter: ffi::Counter,
}

/// Resolved from [`MappingKind::Branch`], and the precursor to [`ffi::BranchRegion`].
#[derive(Debug)]
struct BranchMapping {
    coords: spans::Coords,
    true_counter: ffi::Counter,
    false_counter: ffi::Counter,
}

fn prepare_resolved_mappings<'tcx>(
    tcx: TyCtxt<'tcx>,
    cg_info: &'tcx CoverageCodegenInfo,
    is_used: bool,
    mappings: &[Mapping],
) -> Option<ResolvedMappings> {
    // If this function is unused, replace all counters with zero.
    let counter_for_bcb = |bcb: BasicCoverageBlock| -> ffi::Counter {
        let term = if is_used {
            cg_info.term_for_bcb[bcb].expect("every BCB in a mapping was given a term")
        } else {
            CovTerm::Zero
        };
        counter_for_term(term)
    };

    // Currently a function's mappings must all be in the same file, so use the
    // first mapping's span to determine the file.
    let source_map = tcx.sess.source_map();
    let first_span = mappings.first()?.span;
    let source_file = source_map.lookup_source_file(first_span.lo());

    // In rare cases, _all_ of a function's spans are discarded, and coverage
    // codegen needs to handle that gracefully to avoid #133606.
    // It's hard for tests to trigger this organically, so instead we set
    // `-Zcoverage-options=discard-all-spans-in-codegen` to force it to occur.
    let discard_all = tcx.sess.coverage_options().discard_all_spans_in_codegen;
    let make_coords = |span: Span| {
        if discard_all { None } else { spans::make_coords(source_map, &source_file, span) }
    };

    let mut code_mappings = vec![];
    let mut branch_mappings = vec![];

    for &Mapping { ref kind, span } in mappings {
        let Some(coords) = make_coords(span) else { continue };
        match *kind {
            MappingKind::Code { bcb } => {
                code_mappings.push(CodeMapping { coords, counter: counter_for_bcb(bcb) })
            }
            MappingKind::Branch { true_bcb, false_bcb } => branch_mappings.push(BranchMapping {
                coords,
                true_counter: counter_for_bcb(true_bcb),
                false_counter: counter_for_bcb(false_bcb),
            }),
        }
    }

    ResolvedMappings { source_file, code_mappings, branch_mappings }.ensure_nonempty()
}

/// Populates the mapping region tables for the current function's covfun record.
fn fill_region_tables(
    global_file_table: &GlobalFileTable,
    mappings: &ResolvedMappings,
    virtual_file_mapping: &mut IndexVec<LocalFileId, u32>,
    regions: &mut llvm_cov::Regions,
) {
    let ResolvedMappings { source_file, code_mappings, branch_mappings } = mappings;
    let Some(global_file_id) = global_file_table.get_existing_id(source_file) else {
        debug_assert!(false, "couldn't find an existing global-file-id for {source_file:?}");
        return;
    };

    let llvm_cov::Regions {
        code_regions,
        expansion_regions: _, // FIXME(Zalathar): Fill out support for expansion regions
        branch_regions,
    } = regions;

    // The global file IDs are stored as `u32` to make FFI easier.
    // FIXME(Zalathar): Consider giving `newtype_index!` a safe transmute to `&[u32]`.
    let local_file_id = virtual_file_mapping.push(global_file_id.as_u32());

    for &CodeMapping { coords, counter } in code_mappings {
        let cov_span = coords.make_coverage_span(local_file_id);
        code_regions.push(ffi::CodeRegion { cov_span, counter });
    }

    for &BranchMapping { coords, true_counter, false_counter } in branch_mappings {
        let cov_span = coords.make_coverage_span(local_file_id);
        branch_regions.push(ffi::BranchRegion { cov_span, true_counter, false_counter });
    }
}

/// Generates and emits the covfun record for this function, which
/// contains the function's coverage mapping data. The record is emitted
/// as a global variable in the `__llvm_covfun` section.
pub(crate) fn emit_covfun_record<'tcx>(
    cx: &mut CodegenCx<'_, 'tcx>,
    global_file_table: &GlobalFileTable,
    covfun: &CovfunRecord<'tcx>,
) {
    let &CovfunRecord {
        _instance,
        mangled_function_name,
        source_hash,
        is_used,
        ref expressions,
        ref mappings,
    } = covfun;

    let mut regions = llvm_cov::Regions::default();
    let mut virtual_file_mapping = IndexVec::new();
    fill_region_tables(global_file_table, mappings, &mut virtual_file_mapping, &mut regions);

    if regions.has_no_regions() {
        debug_assert!(false, "mappings should have produced at least one region: {mappings:#?}");
        return;
    }

    // Encode the function's coverage mappings into a buffer.
    let coverage_mapping_buffer = llvm_cov::write_function_mappings_to_buffer(
        &virtual_file_mapping.raw,
        expressions,
        &regions,
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
