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
    pub(crate) function_coverage_map: RefCell<FxHashMap<Instance<'tcx>, FunctionCoverage>>,
}

impl<'tcx> CrateCoverageContext<'tcx> {
    pub fn new() -> Self {
        Self { function_coverage_map: Default::default() }
    }

    pub fn take_function_coverage_map(&self) -> FxHashMap<Instance<'tcx>, FunctionCoverage> {
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
        index: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter to coverage_regions: instance={:?}, function_source_hash={}, index={}, byte range {}..{}",
            instance, function_source_hash, index, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| {
                FunctionCoverage::with_coverageinfo(self.tcx.coverageinfo(instance.def_id()))
            })
            .add_counter(function_source_hash, index, start_byte_pos, end_byte_pos);
    }

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        index: u32,
        lhs: u32,
        op: CounterOp,
        rhs: u32,
        start_byte_pos: u32,
        end_byte_pos: u32,
    ) {
        debug!(
            "adding counter expression to coverage_regions: instance={:?}, index={}, {} {:?} {}, byte range {}..{}",
            instance, index, lhs, op, rhs, start_byte_pos, end_byte_pos,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| {
                FunctionCoverage::with_coverageinfo(self.tcx.coverageinfo(instance.def_id()))
            })
            .add_counter_expression(index, lhs, op, rhs, start_byte_pos, end_byte_pos);
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
            .or_insert_with(|| {
                FunctionCoverage::with_coverageinfo(self.tcx.coverageinfo(instance.def_id()))
            })
            .add_unreachable(start_byte_pos, end_byte_pos);
    }
}

/// This struct wraps an opaque reference to the C++ template instantiation of
/// `llvm::SmallVector<coverage::CounterExpression>`. Each `coverage::CounterExpression` object is
/// constructed from primative-typed arguments, and pushed to the `SmallVector`, in the C++
/// implementation of `LLVMRustCoverageSmallVectorCounterExpressionAdd()` (see
/// `src/rustllvm/CoverageMappingWrapper.cpp`).
pub struct SmallVectorCounterExpression<'a> {
    pub raw: &'a mut llvm::coverageinfo::SmallVectorCounterExpression<'a>,
}

impl SmallVectorCounterExpression<'a> {
    pub fn new() -> Self {
        SmallVectorCounterExpression {
            raw: unsafe { llvm::LLVMRustCoverageSmallVectorCounterExpressionCreate() },
        }
    }

    pub fn as_ptr(&self) -> *const llvm::coverageinfo::SmallVectorCounterExpression<'a> {
        self.raw
    }

    pub fn push_from(
        &mut self,
        kind: rustc_codegen_ssa::coverageinfo::CounterOp,
        left_index: u32,
        right_index: u32,
    ) {
        unsafe {
            llvm::LLVMRustCoverageSmallVectorCounterExpressionAdd(
                &mut *(self.raw as *mut _),
                kind,
                left_index,
                right_index,
            )
        }
    }
}

impl Drop for SmallVectorCounterExpression<'a> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustCoverageSmallVectorCounterExpressionDispose(&mut *(self.raw as *mut _));
        }
    }
}

/// This struct wraps an opaque reference to the C++ template instantiation of
/// `llvm::SmallVector<coverage::CounterMappingRegion>`. Each `coverage::CounterMappingRegion`
/// object is constructed from primative-typed arguments, and pushed to the `SmallVector`, in the
/// C++ implementation of `LLVMRustCoverageSmallVectorCounterMappingRegionAdd()` (see
/// `src/rustllvm/CoverageMappingWrapper.cpp`).
pub struct SmallVectorCounterMappingRegion<'a> {
    pub raw: &'a mut llvm::coverageinfo::SmallVectorCounterMappingRegion<'a>,
}

impl SmallVectorCounterMappingRegion<'a> {
    pub fn new() -> Self {
        SmallVectorCounterMappingRegion {
            raw: unsafe { llvm::LLVMRustCoverageSmallVectorCounterMappingRegionCreate() },
        }
    }

    pub fn as_ptr(&self) -> *const llvm::coverageinfo::SmallVectorCounterMappingRegion<'a> {
        self.raw
    }

    pub fn push_from(
        &mut self,
        index: u32,
        file_id: u32,
        line_start: u32,
        column_start: u32,
        line_end: u32,
        column_end: u32,
    ) {
        unsafe {
            llvm::LLVMRustCoverageSmallVectorCounterMappingRegionAdd(
                &mut *(self.raw as *mut _),
                index,
                file_id,
                line_start,
                column_start,
                line_end,
                column_end,
            )
        }
    }
}

impl Drop for SmallVectorCounterMappingRegion<'a> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustCoverageSmallVectorCounterMappingRegionDispose(
                &mut *(self.raw as *mut _),
            );
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
    expressions: SmallVectorCounterExpression<'_>,
    mapping_regions: SmallVectorCounterMappingRegion<'_>,
    buffer: &RustString,
) {
    unsafe {
        llvm::LLVMRustCoverageWriteMappingToBuffer(
            virtual_file_mapping.as_ptr(),
            virtual_file_mapping.len() as c_uint,
            expressions.as_ptr(),
            mapping_regions.as_ptr(),
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
