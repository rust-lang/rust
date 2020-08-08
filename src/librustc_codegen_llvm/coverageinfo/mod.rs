use crate::llvm;

use crate::builder::Builder;
use crate::common::CodegenCx;

use libc::c_uint;
use llvm::coverageinfo::CounterMappingRegion;
use log::debug;
use rustc_codegen_ssa::coverageinfo::map::{CounterExpression, ExprKind, FunctionCoverage, Region};
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
        region: Region<'tcx>,
    ) {
        debug!(
            "adding counter to coverage_regions: instance={:?}, function_source_hash={}, id={}, \
             at {:?}",
            instance, function_source_hash, id, region,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_counter(function_source_hash, id, region);
    }

    fn add_counter_expression_region(
        &mut self,
        instance: Instance<'tcx>,
        id_descending_from_max: u32,
        lhs: u32,
        op: ExprKind,
        rhs: u32,
        region: Region<'tcx>,
    ) {
        debug!(
            "adding counter expression to coverage_regions: instance={:?}, id={}, {} {:?} {}, \
             at {:?}",
            instance, id_descending_from_max, lhs, op, rhs, region,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_counter_expression(id_descending_from_max, lhs, op, rhs, region);
    }

    fn add_unreachable_region(&mut self, instance: Instance<'tcx>, region: Region<'tcx>) {
        debug!(
            "adding unreachable code to coverage_regions: instance={:?}, at {:?}",
            instance, region,
        );
        let mut coverage_regions = self.coverage_context().function_coverage_map.borrow_mut();
        coverage_regions
            .entry(instance)
            .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
            .add_unreachable_region(region);
    }
}

pub(crate) fn write_filenames_section_to_buffer<'a>(
    filenames: impl IntoIterator<Item = &'a CString>,
    buffer: &RustString,
) {
    let c_str_vec = filenames.into_iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();
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
