use crate::llvm;

use crate::builder::Builder;
use crate::common::CodegenCx;

use libc::c_uint;
use log::debug;
use rustc_codegen_ssa::coverageinfo::map::*;
use rustc_codegen_ssa::traits::{
    BaseTypeMethods, CoverageInfoBuilderMethods, CoverageInfoMethods, StaticMethods,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_llvm::RustString;
use rustc_middle::ty::Instance;

use std::cell::RefCell;
use std::ffi::CString;

pub mod mapgen;

const COVMAP_VAR_ALIGN_BYTES: usize = 8;

/// A context object for maintaining all state needed by the coverageinfo module.
pub struct CrateCoverageContext<'tcx> {
    // Coverage region data for each instrumented function identified by DefId.
    pub(crate) function_coverage_map: RefCell<FxHashMap<Instance<'tcx>, FunctionCoverage<'tcx>>>,
}

impl<'tcx> CrateCoverageContext<'tcx> {
    pub fn new() -> Self {
        Self { function_coverage_map: Default::default() }
    }

    pub fn take_function_coverage_map(&self) -> FxHashMap<Instance<'tcx>, FunctionCoverage<'tcx>> {
        self.function_coverage_map.replace(FxHashMap::default())
    }
}

impl CoverageInfoMethods for CodegenCx<'ll, 'tcx> {
    fn coverageinfo_finalize(&self) {
        mapgen::finalize(self)
    }
}

impl CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'll, 'tcx> {
    fn add_counter_region(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
        id: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter to coverage_regions: instance={:?}, function_source_hash={}, id={}, \
             byte range {}..{}",
            instance, function_source_hash, id, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_counter(function_source_hash, id, start_byte_pos, end_byte_pos);
    }

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        id_descending_from_max: u32,
        lhs: u32,
        op: ExprKind,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter expression to coverage_regions: instance={:?}, id={}, {} {:?} {}, \
             byte range {}..{}",
            instance, id_descending_from_max, lhs, op, rhs, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_counter_expression(
                id_descending_from_max,
                lhs,
                op,
                rhs,
                start_byte_pos,
                end_byte_pos,
            );
    }

    fn add_unreachable_region(
        &mut self,
        instance: Instance<'tcx>,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding unreachable code to coverage_regions: instance={:?}, byte range {}..{}",
            instance, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_unreachable_region(start_byte_pos, end_byte_pos);
    }
}

/// Aligns with [llvm::coverage::CounterMappingRegion::RegionKind](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L205-L221)
#[derive(Copy, Clone, Debug)]
#[repr(C)]
enum RegionKind {
    /// A CodeRegion associates some code with a counter
    CodeRegion = 0,

    /// An ExpansionRegion represents a file expansion region that associates
    /// a source range with the expansion of a virtual source file, such as
    /// for a macro instantiation or #include file.
    ExpansionRegion = 1,

    /// A SkippedRegion represents a source range with code that was skipped
    /// by a preprocessor or similar means.
    SkippedRegion = 2,

    /// A GapRegion is like a CodeRegion, but its count is only set as the
    /// line execution count when its the only region in the line.
    GapRegion = 3,
}

/// This struct provides LLVM's representation of a "CoverageMappingRegion", encoded into the
/// coverage map, in accordance with the
/// [LLVM Code Coverage Mapping Format](https://github.com/rust-lang/llvm-project/blob/llvmorg-8.0.0/llvm/docs/CoverageMappingFormat.rst#llvm-code-coverage-mapping-format).
/// The struct composes fields representing the `Counter` type and value(s) (injected counter ID,
/// or expression type and operands), the source file (an indirect index into a "filenames array",
/// encoded separately), and source location (start and end positions of the represented code
/// region).
///
/// Aligns with [llvm::coverage::CounterMappingRegion](https://github.com/rust-lang/llvm-project/blob/rustc/10.0-2020-05-05/llvm/include/llvm/ProfileData/Coverage/CoverageMapping.h#L223-L226)
/// Important: The Rust struct layout (order and types of fields) must match its C++ counterpart.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CounterMappingRegion {
    /// The counter type and type-dependent counter data, if any.
    counter: Counter,

    /// An indirect reference to the source filename. In the LLVM Coverage Mapping Format, the
    /// file_id is an index into a function-specific `virtual_file_mapping` array of indexes that,
    /// in turn, are used to look up the filename for this region.
    file_id: u32,

    /// If the `RegionKind` is an `ExpansionRegion`, the `expanded_file_id` can be used to find the
    /// mapping regions created as a result of macro expansion, by checking if their file id matches
    /// the expanded file id.
    expanded_file_id: u32,

    /// 1-based starting line of the mapping region.
    start_line: u32,

    /// 1-based starting column of the mapping region.
    start_col: u32,

    /// 1-based ending line of the mapping region.
    end_line: u32,

    /// 1-based ending column of the mapping region. If the high bit is set, the current mapping
    /// region is a gap area.
    end_col: u32,

    kind: RegionKind,
}

impl CounterMappingRegion {
    pub fn code_region(
        counter: Counter,
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::CodeRegion,
        }
    }

    pub fn expansion_region(
        file_id: u32,
        expanded_file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter: Counter::zero(),
            file_id,
            expanded_file_id,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::ExpansionRegion,
        }
    }

    pub fn skipped_region(
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter: Counter::zero(),
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col,
            kind: RegionKind::SkippedRegion,
        }
    }

    pub fn gap_region(
        counter: Counter,
        file_id: u32,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self {
            counter,
            file_id,
            expanded_file_id: 0,
            start_line,
            start_col,
            end_line,
            end_col: ((1 as u32) << 31) | end_col,
            kind: RegionKind::GapRegion,
        }
    }
}

pub(crate) fn write_filenames_section_to_buffer(filenames: &Vec<CString>, buffer: &RustString) {
    let c_str_vec = filenames.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();
    unsafe {
        llvm::LLVMRustCoverageWriteFilenamesSectionToBuffer(
            c_str_vec.as_ptr(),
            c_str_vec.len(),
            buffer,
        );
    }
}

pub(crate) fn write_mapping_to_buffer(
    virtual_file_mapping: Vec<u32>,
    expressions: Vec<CounterExpression>,
    mut mapping_regions: Vec<CounterMappingRegion>,
    buffer: &RustString,
) {
    unsafe {
        llvm::LLVMRustCoverageWriteMappingToBuffer(
            virtual_file_mapping.as_ptr(),
            virtual_file_mapping.len() as c_uint,
            expressions.as_ptr(),
            expressions.len() as c_uint,
            mapping_regions.as_mut_ptr(),
            mapping_regions.len() as c_uint,
            buffer,
        );
    }
}

pub(crate) fn compute_hash(name: &str) -> u64 {
    let name = CString::new(name).expect("null error converting hashable name to C string");
    unsafe { llvm::LLVMRustCoverageComputeHash(name.as_ptr()) }
}

pub(crate) fn mapping_version() -> u32 {
    unsafe { llvm::LLVMRustCoverageMappingVersion() }
}

pub(crate) fn save_map_to_mod<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    cov_data_val: &'ll llvm::Value,
) {
    let covmap_var_name = llvm::build_string(|s| unsafe {
        llvm::LLVMRustCoverageWriteMappingVarNameToString(s);
    })
    .expect("Rust Coverage Mapping var name failed UTF-8 conversion");
    debug!("covmap var name: {:?}", covmap_var_name);

    let covmap_section_name = llvm::build_string(|s| unsafe {
        llvm::LLVMRustCoverageWriteSectionNameToString(cx.llmod, s);
    })
    .expect("Rust Coverage section name failed UTF-8 conversion");
    debug!("covmap section name: {:?}", covmap_section_name);

    let llglobal = llvm::add_global(cx.llmod, cx.val_ty(cov_data_val), &covmap_var_name);
    llvm::set_initializer(llglobal, cov_data_val);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, llvm::Linkage::InternalLinkage);
    llvm::set_section(llglobal, &covmap_section_name);
    llvm::set_alignment(llglobal, COVMAP_VAR_ALIGN_BYTES);
    cx.add_used_global(llglobal);
}
