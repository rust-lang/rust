use crate::llvm;

use crate::abi::Abi;
use crate::builder::Builder;
use crate::common::CodegenCx;

use libc::c_uint;
use llvm::coverageinfo::CounterMappingRegion;
use rustc_codegen_ssa::coverageinfo::map::{CounterExpression, FunctionCoverage};
use rustc_codegen_ssa::traits::{
    BaseTypeMethods, BuilderMethods, ConstMethods, CoverageInfoBuilderMethods, CoverageInfoMethods,
    MiscMethods, StaticMethods,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_llvm::RustString;
use rustc_middle::bug;
use rustc_middle::mir::coverage::{
    CodeRegion, CounterValueReference, ExpressionOperandId, InjectedExpressionId, Op,
};
use rustc_middle::ty;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::Instance;

use std::cell::RefCell;
use std::ffi::CString;

use std::iter;

pub mod mapgen;

const UNUSED_FUNCTION_COUNTER_ID: CounterValueReference = CounterValueReference::START;

const VAR_ALIGN_BYTES: usize = 8;

/// A context object for maintaining all state needed by the coverageinfo module.
pub struct CrateCoverageContext<'ll, 'tcx> {
    // Coverage data for each instrumented function identified by DefId.
    pub(crate) function_coverage_map: RefCell<FxHashMap<Instance<'tcx>, FunctionCoverage<'tcx>>>,
    pub(crate) pgo_func_name_var_map: RefCell<FxHashMap<Instance<'tcx>, &'ll llvm::Value>>,
}

impl<'ll, 'tcx> CrateCoverageContext<'ll, 'tcx> {
    pub fn new() -> Self {
        Self {
            function_coverage_map: Default::default(),
            pgo_func_name_var_map: Default::default(),
        }
    }

    pub fn take_function_coverage_map(&self) -> FxHashMap<Instance<'tcx>, FunctionCoverage<'tcx>> {
        self.function_coverage_map.replace(FxHashMap::default())
    }
}

impl<'ll, 'tcx> CoverageInfoMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn coverageinfo_finalize(&self) {
        mapgen::finalize(self)
    }

    fn get_pgo_func_name_var(&self, instance: Instance<'tcx>) -> &'ll llvm::Value {
        if let Some(coverage_context) = self.coverage_context() {
            debug!("getting pgo_func_name_var for instance={:?}", instance);
            let mut pgo_func_name_var_map = coverage_context.pgo_func_name_var_map.borrow_mut();
            pgo_func_name_var_map
                .entry(instance)
                .or_insert_with(|| create_pgo_func_name_var(self, instance))
        } else {
            bug!("Could not get the `coverage_context`");
        }
    }

    /// Functions with MIR-based coverage are normally codegenned _only_ if
    /// called. LLVM coverage tools typically expect every function to be
    /// defined (even if unused), with at least one call to LLVM intrinsic
    /// `instrprof.increment`.
    ///
    /// Codegen a small function that will never be called, with one counter
    /// that will never be incremented.
    ///
    /// For used/called functions, the coverageinfo was already added to the
    /// `function_coverage_map` (keyed by function `Instance`) during codegen.
    /// But in this case, since the unused function was _not_ previously
    /// codegenned, collect the coverage `CodeRegion`s from the MIR and add
    /// them. The first `CodeRegion` is used to add a single counter, with the
    /// same counter ID used in the injected `instrprof.increment` intrinsic
    /// call. Since the function is never called, all other `CodeRegion`s can be
    /// added as `unreachable_region`s.
    fn define_unused_fn(&self, def_id: DefId) {
        let instance = declare_unused_fn(self, def_id);
        codegen_unused_fn_and_counter(self, instance);
        add_unused_function_coverage(self, instance, def_id);
    }
}

impl<'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'_, '_, 'tcx> {
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

fn declare_unused_fn<'tcx>(cx: &CodegenCx<'_, 'tcx>, def_id: DefId) -> Instance<'tcx> {
    let tcx = cx.tcx;

    let instance = Instance::new(
        def_id,
        InternalSubsts::for_item(tcx, def_id, |param, _| {
            if let ty::GenericParamDefKind::Lifetime = param.kind {
                tcx.lifetimes.re_erased.into()
            } else {
                tcx.mk_param_from_def(param)
            }
        }),
    );

    let llfn = cx.declare_fn(
        tcx.symbol_name(instance).name,
        cx.fn_abi_of_fn_ptr(
            ty::Binder::dummy(tcx.mk_fn_sig(
                iter::once(tcx.mk_unit()),
                tcx.mk_unit(),
                false,
                hir::Unsafety::Unsafe,
                Abi::Rust,
            )),
            ty::List::empty(),
        ),
    );

    llvm::set_linkage(llfn, llvm::Linkage::PrivateLinkage);
    llvm::set_visibility(llfn, llvm::Visibility::Default);

    assert!(cx.instances.borrow_mut().insert(instance, llfn).is_none());

    instance
}

fn codegen_unused_fn_and_counter<'tcx>(cx: &CodegenCx<'_, 'tcx>, instance: Instance<'tcx>) {
    let llfn = cx.get_fn(instance);
    let llbb = Builder::append_block(cx, llfn, "unused_function");
    let mut bx = Builder::build(cx, llbb);
    let fn_name = bx.get_pgo_func_name_var(instance);
    let hash = bx.const_u64(0);
    let num_counters = bx.const_u32(1);
    let index = bx.const_u32(u32::from(UNUSED_FUNCTION_COUNTER_ID));
    debug!(
        "codegen intrinsic instrprof.increment(fn_name={:?}, hash={:?}, num_counters={:?},
            index={:?}) for unused function: {:?}",
        fn_name, hash, num_counters, index, instance
    );
    bx.instrprof_increment(fn_name, hash, num_counters, index);
    bx.ret_void();
}

fn add_unused_function_coverage<'tcx>(
    cx: &CodegenCx<'_, 'tcx>,
    instance: Instance<'tcx>,
    def_id: DefId,
) {
    let tcx = cx.tcx;

    let mut function_coverage = FunctionCoverage::unused(tcx, instance);
    for (index, &code_region) in tcx.covered_code_regions(def_id).iter().enumerate() {
        if index == 0 {
            // Insert at least one real counter so the LLVM CoverageMappingReader will find expected
            // definitions.
            function_coverage.add_counter(UNUSED_FUNCTION_COUNTER_ID, code_region.clone());
        } else {
            function_coverage.add_unreachable_region(code_region.clone());
        }
    }

    if let Some(coverage_context) = cx.coverage_context() {
        coverage_context.function_coverage_map.borrow_mut().insert(instance, function_coverage);
    } else {
        bug!("Could not get the `coverage_context`");
    }
}

/// Calls llvm::createPGOFuncNameVar() with the given function instance's
/// mangled function name. The LLVM API returns an llvm::GlobalVariable
/// containing the function name, with the specific variable name and linkage
/// required by LLVM InstrProf source-based coverage instrumentation. Use
/// `bx.get_pgo_func_name_var()` to ensure the variable is only created once per
/// `Instance`.
fn create_pgo_func_name_var<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    instance: Instance<'tcx>,
) -> &'ll llvm::Value {
    let mangled_fn_name = CString::new(cx.tcx.symbol_name(instance).name)
        .expect("error converting function name to C string");
    let llfn = cx.get_fn(instance);
    unsafe { llvm::LLVMRustCoverageCreatePGOFuncNameVar(llfn, mangled_fn_name.as_ptr()) }
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
    mapping_regions: Vec<CounterMappingRegion>,
    buffer: &RustString,
) {
    unsafe {
        llvm::LLVMRustCoverageWriteMappingToBuffer(
            virtual_file_mapping.as_ptr(),
            virtual_file_mapping.len() as c_uint,
            expressions.as_ptr(),
            expressions.len() as c_uint,
            mapping_regions.as_ptr(),
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
