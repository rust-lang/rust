//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![feature(extern_types)]
#![feature(file_buffered)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(macro_derive)]
#![feature(once_cell_try)]
#![feature(trim_prefix_suffix)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use std::any::Any;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::path::PathBuf;

use back::owned_target_machine::OwnedTargetMachine;
use back::write::{create_informational_target_machine, create_target_machine};
use context::SimpleCx;
use llvm_util::target_config;
use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_codegen_ssa::back::lto::{SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, SharedEmitter, TargetMachineFactoryConfig,
    TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::{DiagCtxt, DiagCtxtHandle};
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames, PrintKind, PrintRequest};
use rustc_span::{Symbol, sym};
use rustc_target::spec::{RelocModel, TlsModel};

use crate::llvm::ToLlvmBool;

mod abi;
mod allocator;
mod asm;
mod attributes;
mod back;
mod base;
mod builder;
mod callee;
mod common;
mod consts;
mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod errors;
mod intrinsic;
mod llvm;
mod llvm_util;
mod macros;
mod mono_item;
mod type_;
mod type_of;
mod typetree;
mod va_arg;
mod value;

pub(crate) use macros::TryFromU32;

#[derive(Clone)]
pub struct LlvmCodegenBackend(());

struct TimeTraceProfiler {}

impl TimeTraceProfiler {
    fn new() -> Self {
        unsafe { llvm::LLVMRustTimeTraceProfilerInitialize() }
        TimeTraceProfiler {}
    }
}

impl Drop for TimeTraceProfiler {
    fn drop(&mut self) {
        unsafe { llvm::LLVMRustTimeTraceProfilerFinishThread() }
    }
}

impl ExtraBackendMethods for LlvmCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        methods: &[AllocatorMethod],
    ) -> ModuleLlvm {
        let module_llvm = ModuleLlvm::new_metadata(tcx, module_name);
        let cx =
            SimpleCx::new(module_llvm.llmod(), &module_llvm.llcx, tcx.data_layout.pointer_size());
        unsafe {
            allocator::codegen(tcx, cx, module_name, methods);
        }
        module_llvm
    }
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<ModuleLlvm>, u64) {
        base::compile_codegen_unit(tcx, cgu_name)
    }
}

impl WriteBackendMethods for LlvmCodegenBackend {
    type Module = ModuleLlvm;
    type ModuleBuffer = back::lto::ModuleBuffer;
    type TargetMachine = OwnedTargetMachine;
    type ThinData = back::lto::ThinData;
    fn thread_profiler() -> Box<dyn Any> {
        Box::new(TimeTraceProfiler::new())
    }
    fn target_machine_factory(
        &self,
        sess: &Session,
        optlvl: OptLevel,
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        back::write::target_machine_factory(sess, optlvl, target_features)
    }
    fn optimize_and_codegen_fat_lto(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        tm_factory: TargetMachineFactoryFn<LlvmCodegenBackend>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule {
        let mut module = back::lto::run_fat(
            cgcx,
            prof,
            shared_emitter,
            tm_factory,
            exported_symbols_for_lto,
            each_linked_rlib_for_lto,
            modules,
        );

        let dcx = DiagCtxt::new(Box::new(shared_emitter.clone()));
        let dcx = dcx.handle();
        back::lto::run_pass_manager(cgcx, prof, dcx, &mut module, false);

        back::write::codegen(cgcx, prof, shared_emitter, module, &cgcx.module_config)
    }
    fn run_thin_lto(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        dcx: DiagCtxtHandle<'_>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ModuleBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        back::lto::run_thin(
            cgcx,
            prof,
            dcx,
            exported_symbols_for_lto,
            each_linked_rlib_for_lto,
            modules,
            cached_modules,
        )
    }
    fn optimize(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) {
        back::write::optimize(cgcx, prof, shared_emitter, module, config)
    }
    fn optimize_and_codegen_thin(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        tm_factory: TargetMachineFactoryFn<LlvmCodegenBackend>,
        thin: ThinModule<Self>,
    ) -> CompiledModule {
        back::lto::optimize_and_codegen_thin_module(cgcx, prof, shared_emitter, tm_factory, thin)
    }
    fn codegen(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule {
        back::write::codegen(cgcx, prof, shared_emitter, module, config)
    }
    fn serialize_module(module: Self::Module, is_thin: bool) -> Self::ModuleBuffer {
        back::lto::ModuleBuffer::new(module.llmod(), is_thin)
    }
}

impl LlvmCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(LlvmCodegenBackend(()))
    }
}

impl CodegenBackend for LlvmCodegenBackend {
    fn name(&self) -> &'static str {
        "llvm"
    }

    fn init(&self, sess: &Session) {
        llvm_util::init(sess); // Make sure llvm is inited

        // autodiff is based on Enzyme, a library which we might not have available, when it was
        // neither build, nor downloaded via rustup. If autodiff is used, but not available we emit
        // an early error here and abort compilation.
        {
            use rustc_session::config::AutoDiff;

            use crate::back::lto::enable_autodiff_settings;
            if sess.opts.unstable_opts.autodiff.contains(&AutoDiff::Enable) {
                match llvm::EnzymeWrapper::get_or_init(&sess.opts.sysroot) {
                    Ok(_) => {}
                    Err(llvm::EnzymeLibraryError::NotFound { err }) => {
                        sess.dcx().emit_fatal(crate::errors::AutoDiffComponentMissing { err });
                    }
                    Err(llvm::EnzymeLibraryError::LoadFailed { err }) => {
                        sess.dcx().emit_fatal(crate::errors::AutoDiffComponentUnavailable { err });
                    }
                }
                enable_autodiff_settings(&sess.opts.unstable_opts.autodiff);
            }
        }
    }

    fn provide(&self, providers: &mut Providers) {
        providers.queries.global_backend_features =
            |tcx, ()| llvm_util::global_llvm_features(tcx.sess, false)
    }

    fn print(&self, req: &PrintRequest, out: &mut String, sess: &Session) {
        use std::fmt::Write;
        match req.kind {
            PrintKind::RelocationModels => {
                writeln!(out, "Available relocation models:").unwrap();
                for name in RelocModel::ALL.iter().map(RelocModel::desc).chain(["default"]) {
                    writeln!(out, "    {name}").unwrap();
                }
                writeln!(out).unwrap();
            }
            PrintKind::CodeModels => {
                writeln!(out, "Available code models:").unwrap();
                for name in &["tiny", "small", "kernel", "medium", "large"] {
                    writeln!(out, "    {name}").unwrap();
                }
                writeln!(out).unwrap();
            }
            PrintKind::TlsModels => {
                writeln!(out, "Available TLS models:").unwrap();
                for name in TlsModel::ALL.iter().map(TlsModel::desc) {
                    writeln!(out, "    {name}").unwrap();
                }
                writeln!(out).unwrap();
            }
            PrintKind::StackProtectorStrategies => {
                writeln!(
                    out,
                    r#"Available stack protector strategies:
    all
        Generate stack canaries in all functions.

    strong
        Generate stack canaries for all functions, unless the compiler
        can prove these functions can't be the source of a stack
        buffer overflow (even in the presence of undefined behavior).

        This provides the same security guarantees as Clang's
        `-fstack-protector=strong`.

        The exact rules are unstable and subject to change, but
        currently, it generates stack protectors for functions that,
        *post-optimization*, contain either arrays (of any size
        or type) or address-taken locals.

    basic
        Generate stack canaries in functions that are heuristically
        suspected to contain buffer overflows.

        The heuristic is subject to change, but currently it
        includes functions with local variables of `[T; N]`
        type, where `T` is byte-sized and `N` >= 8.

        This heuristic originated from C, where it detects
        functions that allocate a `char buf[N];` buffer on the
        stack, and are therefore likely to have a stack buffer overflow
        in the case of a length-calculation error. It is *not* a good
        heuristic for Rust code.

    none
        Do not generate stack canaries.
"#
                )
                .unwrap();
            }
            _other => llvm_util::print(req, out, sess),
        }
    }

    fn print_passes(&self) {
        llvm_util::print_passes();
    }

    fn print_version(&self) {
        llvm_util::print_version();
    }

    fn has_zstd(&self) -> bool {
        llvm::LLVMRustLLVMHasZstdCompression()
    }

    fn target_config(&self, sess: &Session) -> TargetConfig {
        target_config(sess)
    }

    fn replaced_intrinsics(&self) -> Vec<Symbol> {
        let mut will_not_use_fallback =
            vec![sym::unchecked_funnel_shl, sym::unchecked_funnel_shr, sym::carrying_mul_add];

        if llvm_util::get_version() >= (22, 0, 0) {
            will_not_use_fallback.push(sym::carryless_mul);
        }

        will_not_use_fallback
    }

    fn target_cpu(&self, sess: &Session) -> String {
        crate::llvm_util::target_cpu(sess).to_string()
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>, crate_info: &CrateInfo) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(LlvmCodegenBackend(()), tcx, crate_info))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        let (compiled_modules, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<LlvmCodegenBackend>>()
            .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        if sess.opts.unstable_opts.llvm_time_trace {
            sess.time("llvm_dump_timing_file", || {
                let file_name = outputs.with_extension("llvm_timings.json");
                llvm_util::time_trace_profiler_finish(&file_name);
            });
        }

        (compiled_modules, work_products)
    }

    fn print_pass_timings(&self) {
        let timings = llvm::build_string(|s| unsafe { llvm::LLVMRustPrintPassTimings(s) }).unwrap();
        print!("{timings}");
    }

    fn print_statistics(&self) {
        let stats = llvm::build_string(|s| unsafe { llvm::LLVMRustPrintStatistics(s) }).unwrap();
        print!("{stats}");
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        use rustc_codegen_ssa::back::link::link_binary;

        use crate::back::archive::LlvmArchiveBuilderBuilder;

        // Run the linker on any artifacts that resulted from the LLVM run.
        // This should produce either a finished executable or library.
        link_binary(
            sess,
            &LlvmArchiveBuilderBuilder,
            compiled_modules,
            crate_info,
            metadata,
            outputs,
            self.name(),
        );
    }
}

pub struct ModuleLlvm {
    llcx: &'static mut llvm::Context,
    llmod_raw: *const llvm::Module,

    // This field is `ManuallyDrop` because it is important that the `TargetMachine`
    // is disposed prior to the `Context` being disposed otherwise UAFs can occur.
    tm: ManuallyDrop<OwnedTargetMachine>,
}

unsafe impl Send for ModuleLlvm {}
unsafe impl Sync for ModuleLlvm {}

impl ModuleLlvm {
    fn new(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            llvm::LLVMContextSetDiscardValueNames(llcx, tcx.sess.fewer_names().to_llvm_bool());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm {
                llmod_raw,
                llcx,
                tm: ManuallyDrop::new(create_target_machine(tcx, mod_name)),
            }
        }
    }

    fn new_metadata(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            llvm::LLVMContextSetDiscardValueNames(llcx, tcx.sess.fewer_names().to_llvm_bool());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm {
                llmod_raw,
                llcx,
                tm: ManuallyDrop::new(create_informational_target_machine(tcx.sess, false)),
            }
        }
    }

    fn parse(
        cgcx: &CodegenContext,
        tm_factory: TargetMachineFactoryFn<LlvmCodegenBackend>,
        name: &CStr,
        buffer: &[u8],
        dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        unsafe {
            let llcx = llvm::LLVMContextCreate();
            llvm::LLVMContextSetDiscardValueNames(llcx, cgcx.fewer_names.to_llvm_bool());
            let llmod_raw = back::lto::parse_module(llcx, name, buffer, dcx);
            let tm = tm_factory(dcx, TargetMachineFactoryConfig::new(cgcx, name.to_str().unwrap()));

            ModuleLlvm { llmod_raw, llcx, tm: ManuallyDrop::new(tm) }
        }
    }

    fn llmod(&self) -> &llvm::Module {
        unsafe { &*self.llmod_raw }
    }
}

impl Drop for ModuleLlvm {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.tm);
            llvm::LLVMContextDispose(&mut *(self.llcx as *mut _));
        }
    }
}
