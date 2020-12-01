use crate::llvm;

use crate::builder::Builder;
use crate::common::CodegenCx;

use libc::c_uint;
use llvm::coverageinfo::CounterMappingRegion;
use rustc_codegen_ssa::coverageinfo::map::{CounterExpression, FunctionCoverage};
use rustc_codegen_ssa::traits::{
    BaseTypeMethods, CoverageInfoBuilderMethods, CoverageInfoMethods, MiscMethods, StaticMethods,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_llvm::RustString;
use rustc_middle::mir::coverage::{
    CodeRegion, CounterValueReference, ExpressionOperandId, InjectedExpressionId, Op,
};
use rustc_middle::ty::Instance;

use std::cell::RefCell;
use std::ffi::CString;

use tracing::debug;

pub mod mapgen;

const VAR_ALIGN_BYTES: usize = 8;

/// A context object for maintaining all state needed by the coverageinfo module.
pub struct CrateCoverageContext<'tcx> {
    // Coverage data for each instrumented function identified by DefId.
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
    /// Calls llvm::createPGOFuncNameVar() with the given function instance's mangled function name.
    /// The LLVM API returns an llvm::GlobalVariable containing the function name, with the specific
    /// variable name and linkage required by LLVM InstrProf source-based coverage instrumentation.
    fn create_pgo_func_name_var(&self, instance: Instance<'tcx>) -> Self::Value {
        let llfn = self.cx.get_fn(instance);
        let mangled_fn_name = CString::new(self.tcx.symbol_name(instance).name)
            .expect("error converting function name to C string");
        unsafe { llvm::LLVMRustCoverageCreatePGOFuncNameVar(llfn, mangled_fn_name.as_ptr()) }
    }

    fn set_function_source_hash(
        &mut self,
        instance: Instance<'tcx>,
        function_source_hash: u64,
    ) -> bool {
        if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "ensuring function source hash is set for instance={:?}; function_source_hash={}",
                instance, function_source_hash,
            );
            let mut coverage_map = coverage_context.function_coverage_map.borrow_mut();
            coverage_map
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .set_function_source_hash(function_source_hash);
            true
        } else {
            false
        }
    }

    fn add_coverage_counter(
        &mut self,
        instance: Instance<'tcx>,
        id: CounterValueReference,
        region: CodeRegion,
    ) -> bool {
        if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding counter to coverage_map: instance={:?}, id={:?}, region={:?}",
                instance, id, region,
            );
            let mut coverage_map = coverage_context.function_coverage_map.borrow_mut();
            coverage_map
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_counter(id, region);
            true
        } else {
            false
        }
    }

    fn add_coverage_counter_expression(
        &mut self,
        instance: Instance<'tcx>,
        id: InjectedExpressionId,
        lhs: ExpressionOperandId,
        op: Op,
        rhs: ExpressionOperandId,
        region: Option<CodeRegion>,
    ) -> bool {
        if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding counter expression to coverage_map: instance={:?}, id={:?}, {:?} {:?} {:?}; \
                region: {:?}",
                instance, id, lhs, op, rhs, region,
            );
            let mut coverage_map = coverage_context.function_coverage_map.borrow_mut();
            coverage_map
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_counter_expression(id, lhs, op, rhs, region);
            true
        } else {
            false
        }
    }

    fn add_coverage_unreachable(&mut self, instance: Instance<'tcx>, region: CodeRegion) -> bool {
        if let Some(coverage_context) = self.coverage_context() {
            debug!(
                "adding unreachable code to coverage_map: instance={:?}, at {:?}",
                instance, region,
            );
            let mut coverage_map = coverage_context.function_coverage_map.borrow_mut();
            coverage_map
                .entry(instance)
                .or_insert_with(|| FunctionCoverage::new(self.tcx, instance))
                .add_unreachable_region(region);
            true
        } else {
            false
        }
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
pub(crate) fn hash_str(strval: &str) -> u64 {
    let strval = CString::new(strval).expect("null error converting hashable str to C string");
    unsafe { llvm::LLVMRustCoverageHashCString(strval.as_ptr()) }
}

pub(crate) fn hash_bytes(bytes: Vec<u8>) -> u64 {
    unsafe { llvm::LLVMRustCoverageHashByteArray(bytes.as_ptr().cast(), bytes.len()) }
}

pub(crate) fn mapping_version() -> u32 {
    unsafe { llvm::LLVMRustCoverageMappingVersion() }
}

pub(crate) fn save_cov_data_to_mod<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    cov_data_val: &'ll llvm::Value,
) {
    let covmap_var_name = llvm::build_string(|s| unsafe {
        llvm::LLVMRustCoverageWriteMappingVarNameToString(s);
    })
    .expect("Rust Coverage Mapping var name failed UTF-8 conversion");
    debug!("covmap var name: {:?}", covmap_var_name);

    let covmap_section_name = llvm::build_string(|s| unsafe {
        llvm::LLVMRustCoverageWriteMapSectionNameToString(cx.llmod, s);
    })
    .expect("Rust Coverage section name failed UTF-8 conversion");
    debug!("covmap section name: {:?}", covmap_section_name);

    let llglobal = llvm::add_global(cx.llmod, cx.val_ty(cov_data_val), &covmap_var_name);
    llvm::set_initializer(llglobal, cov_data_val);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, llvm::Linkage::PrivateLinkage);
    llvm::set_section(llglobal, &covmap_section_name);
    llvm::set_alignment(llglobal, VAR_ALIGN_BYTES);
    cx.add_used_global(llglobal);
}

pub(crate) fn save_func_record_to_mod<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    func_name_hash: u64,
    func_record_val: &'ll llvm::Value,
    is_used: bool,
) {
    // Assign a name to the function record. This is used to merge duplicates.
    //
    // In LLVM, a "translation unit" (effectively, a `Crate` in Rust) can describe functions that
    // are included-but-not-used. If (or when) Rust generates functions that are
    // included-but-not-used, note that a dummy description for a function included-but-not-used
    // in a Crate can be replaced by full description provided by a different Crate. The two kinds
    // of descriptions play distinct roles in LLVM IR; therefore, assign them different names (by
    // appending "u" to the end of the function record var name, to prevent `linkonce_odr` merging.
    let func_record_var_name =
        format!("__covrec_{:X}{}", func_name_hash, if is_used { "u" } else { "" });
    debug!("function record var name: {:?}", func_record_var_name);

    let func_record_section_name = llvm::build_string(|s| unsafe {
        llvm::LLVMRustCoverageWriteFuncSectionNameToString(cx.llmod, s);
    })
    .expect("Rust Coverage function record section name failed UTF-8 conversion");
    debug!("function record section name: {:?}", func_record_section_name);

    let llglobal = llvm::add_global(cx.llmod, cx.val_ty(func_record_val), &func_record_var_name);
    llvm::set_initializer(llglobal, func_record_val);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, llvm::Linkage::LinkOnceODRLinkage);
    llvm::set_visibility(llglobal, llvm::Visibility::Hidden);
    llvm::set_section(llglobal, &func_record_section_name);
    llvm::set_alignment(llglobal, VAR_ALIGN_BYTES);
    llvm::set_comdat(cx.llmod, llglobal, &func_record_var_name);
    cx.add_used_global(llglobal);
}
