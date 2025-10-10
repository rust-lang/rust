//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(extern_types)]
#![feature(file_buffered)]
#![feature(if_let_guard)]
#![feature(impl_trait_in_assoc_type)]
#![feature(iter_intersperse)]
#![feature(macro_derive)]
#![feature(rustdoc_internals)]
#![feature(slice_as_array)]
#![feature(try_blocks)]
// tidy-alphabetical-end

use std::any::Any;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::path::PathBuf;

use back::owned_target_machine::OwnedTargetMachine;
use back::write::{create_informational_target_machine, create_target_machine};
use context::SimpleCx;
use errors::ParseTargetMachineConfig;
use llvm_util::target_config;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_codegen_ssa::back::lto::{SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, TargetMachineFactoryConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::DiagCtxtHandle;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames, PrintKind, PrintRequest};
use rustc_span::Symbol;
use rustc_target::spec::{RelocModel, TlsModel};

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

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub(crate) use macros::TryFromU32;

#[derive(Clone)]
pub struct LlvmCodegenBackend(());

struct TimeTraceProfiler {
    enabled: bool,
}

impl TimeTraceProfiler {
    fn new(enabled: bool) -> Self {
        if enabled {
            unsafe { llvm::LLVMRustTimeTraceProfilerInitialize() }
        }
        TimeTraceProfiler { enabled }
    }
}

impl Drop for TimeTraceProfiler {
    fn drop(&mut self) {
        if self.enabled {
            unsafe { llvm::LLVMRustTimeTraceProfilerFinishThread() }
        }
    }
}

impl ExtraBackendMethods for LlvmCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        kind: AllocatorKind,
        alloc_error_handler_kind: AllocatorKind,
    ) -> ModuleLlvm {
        let module_llvm = ModuleLlvm::new_metadata(tcx, module_name);
        let cx =
            SimpleCx::new(module_llvm.llmod(), &module_llvm.llcx, tcx.data_layout.pointer_size());
        unsafe {
            allocator::codegen(tcx, cx, module_name, kind, alloc_error_handler_kind);
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
    fn target_machine_factory(
        &self,
        sess: &Session,
        optlvl: OptLevel,
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        back::write::target_machine_factory(sess, optlvl, target_features)
    }

    fn spawn_named_thread<F, T>(
        time_trace: bool,
        name: String,
        f: F,
    ) -> std::io::Result<std::thread::JoinHandle<T>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        std::thread::Builder::new().name(name).spawn(move || {
            let _profiler = TimeTraceProfiler::new(time_trace);
            f()
        })
    }
}

impl WriteBackendMethods for LlvmCodegenBackend {
    type Module = ModuleLlvm;
    type ModuleBuffer = back::lto::ModuleBuffer;
    type TargetMachine = OwnedTargetMachine;
    type TargetMachineError = crate::errors::LlvmError<'static>;
    type ThinData = back::lto::ThinData;
    type ThinBuffer = back::lto::ThinBuffer;
    fn print_pass_timings(&self) {
        let timings = llvm::build_string(|s| unsafe { llvm::LLVMRustPrintPassTimings(s) }).unwrap();
        print!("{timings}");
    }
    fn print_statistics(&self) {
        let stats = llvm::build_string(|s| unsafe { llvm::LLVMRustPrintStatistics(s) }).unwrap();
        print!("{stats}");
    }
    fn run_and_optimize_fat_lto(
        cgcx: &CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module> {
        let mut module =
            back::lto::run_fat(cgcx, exported_symbols_for_lto, each_linked_rlib_for_lto, modules);

        let dcx = cgcx.create_dcx();
        let dcx = dcx.handle();
        back::lto::run_pass_manager(cgcx, dcx, &mut module, false);

        module
    }
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        back::lto::run_thin(
            cgcx,
            exported_symbols_for_lto,
            each_linked_rlib_for_lto,
            modules,
            cached_modules,
        )
    }
    fn optimize(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) {
        back::write::optimize(cgcx, dcx, module, config)
    }
    fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> ModuleCodegen<Self::Module> {
        back::lto::optimize_thin_module(thin, cgcx)
    }
    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule {
        back::write::codegen(cgcx, module, config)
    }
    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        back::lto::prepare_thin(module)
    }
    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        (module.name, back::lto::ModuleBuffer::new(module.module_llvm.llmod()))
    }
}

impl LlvmCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(LlvmCodegenBackend(()))
    }
}

impl CodegenBackend for LlvmCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        crate::DEFAULT_LOCALE_RESOURCE
    }

    fn name(&self) -> &'static str {
        "llvm"
    }

    fn init(&self, sess: &Session) {
        llvm_util::init(sess); // Make sure llvm is inited
    }

    fn provide(&self, providers: &mut Providers) {
        providers.global_backend_features =
            |tcx, ()| llvm_util::global_llvm_features(tcx.sess, true, false)
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
        Generate stack canaries in a function if it either:
        - has a local variable of `[T; N]` type, regardless of `T` and `N`
        - takes the address of a local variable.

          (Note that a local variable being borrowed is not equivalent to its
          address being taken: e.g. some borrows may be removed by optimization,
          while by-value argument passing may be implemented with reference to a
          local stack variable in the ABI.)

    basic
        Generate stack canaries in functions with local variables of `[T; N]`
        type, where `T` is byte-sized and `N` >= 8.

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

    fn target_config(&self, sess: &Session) -> TargetConfig {
        target_config(sess)
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            LlvmCodegenBackend(()),
            tcx,
            crate::llvm_util::target_cpu(tcx.sess).to_string(),
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<LlvmCodegenBackend>>()
            .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        if sess.opts.unstable_opts.llvm_time_trace {
            sess.time("llvm_dump_timing_file", || {
                let file_name = outputs.with_extension("llvm_timings.json");
                llvm_util::time_trace_profiler_finish(&file_name);
            });
        }

        (codegen_results, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
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
            codegen_results,
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
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
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
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm {
                llmod_raw,
                llcx,
                tm: ManuallyDrop::new(create_informational_target_machine(tcx.sess, false)),
            }
        }
    }

    fn tm_from_cgcx(
        cgcx: &CodegenContext<LlvmCodegenBackend>,
        name: &str,
        dcx: DiagCtxtHandle<'_>,
    ) -> OwnedTargetMachine {
        let tm_factory_config = TargetMachineFactoryConfig::new(cgcx, name);
        match (cgcx.tm_factory)(tm_factory_config) {
            Ok(m) => m,
            Err(e) => {
                dcx.emit_fatal(ParseTargetMachineConfig(e));
            }
        }
    }

    fn parse(
        cgcx: &CodegenContext<LlvmCodegenBackend>,
        name: &CStr,
        buffer: &[u8],
        dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
            let llmod_raw = back::lto::parse_module(llcx, name, buffer, dcx);
            let tm = ModuleLlvm::tm_from_cgcx(cgcx, name.to_str().unwrap(), dcx);

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
