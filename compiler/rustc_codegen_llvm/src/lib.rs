//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(const_cstr_unchecked)]
#![feature(crate_visibility_modifier)]
#![feature(extern_types)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(or_patterns)]
#![recursion_limit = "256"]

use back::write::{create_informational_target_machine, create_target_machine};

pub use llvm_util::target_features;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLTOInput, ModuleConfig, TargetMachineFactoryConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_ssa::{CodegenResults, CompiledModule};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{ErrorReported, FatalError, Handler};
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::cstore::{EncodedMetadata, MetadataLoaderDyn};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::{OptLevel, OutputFilenames, PrintRequest};
use rustc_session::Session;
use rustc_span::symbol::Symbol;

use std::any::Any;
use std::ffi::CStr;

mod back {
    pub mod archive;
    pub mod lto;
    mod profiling;
    pub mod write;
}

mod abi;
mod allocator;
mod asm;
mod attributes;
mod base;
mod builder;
mod callee;
mod common;
mod consts;
mod context;
mod coverageinfo;
mod debuginfo;
mod declare;
mod intrinsic;

// The following is a work around that replaces `pub mod llvm;` and that fixes issue 53912.
#[path = "llvm/mod.rs"]
mod llvm_;
pub mod llvm {
    pub use super::llvm_::*;
}

mod llvm_util;
mod metadata;
mod mono_item;
mod type_;
mod type_of;
mod va_arg;
mod value;

#[derive(Clone)]
pub struct LlvmCodegenBackend(());

impl ExtraBackendMethods for LlvmCodegenBackend {
    fn new_metadata(&self, tcx: TyCtxt<'_>, mod_name: &str) -> ModuleLlvm {
        ModuleLlvm::new_metadata(tcx, mod_name)
    }

    fn write_compressed_metadata<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: &EncodedMetadata,
        llvm_module: &mut ModuleLlvm,
    ) {
        base::write_compressed_metadata(tcx, metadata, llvm_module)
    }
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        mods: &mut ModuleLlvm,
        kind: AllocatorKind,
        has_alloc_error_handler: bool,
    ) {
        unsafe { allocator::codegen(tcx, mods, kind, has_alloc_error_handler) }
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
    ) -> TargetMachineFactoryFn<Self> {
        back::write::target_machine_factory(sess, optlvl)
    }
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str {
        llvm_util::target_cpu(sess)
    }
    fn tune_cpu<'b>(&self, sess: &'b Session) -> Option<&'b str> {
        llvm_util::tune_cpu(sess)
    }
}

impl WriteBackendMethods for LlvmCodegenBackend {
    type Module = ModuleLlvm;
    type ModuleBuffer = back::lto::ModuleBuffer;
    type Context = llvm::Context;
    type TargetMachine = &'static mut llvm::TargetMachine;
    type ThinData = back::lto::ThinData;
    type ThinBuffer = back::lto::ThinBuffer;
    fn print_pass_timings(&self) {
        unsafe {
            llvm::LLVMRustPrintPassTimings();
        }
    }
    fn run_link(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        modules: Vec<ModuleCodegen<Self::Module>>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::write::link(cgcx, diag_handler, modules)
    }
    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLTOInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<LtoModuleCodegen<Self>, FatalError> {
        back::lto::run_fat(cgcx, modules, cached_modules)
    }
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        back::lto::run_thin(cgcx, modules, cached_modules)
    }
    unsafe fn optimize(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        Ok(back::write::optimize(cgcx, diag_handler, module, config))
    }
    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: &mut ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::lto::optimize_thin_module(thin, cgcx)
    }
    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<CompiledModule, FatalError> {
        back::write::codegen(cgcx, diag_handler, module, config)
    }
    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        back::lto::prepare_thin(module)
    }
    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        (module.name, back::lto::ModuleBuffer::new(module.module_llvm.llmod()))
    }
    fn run_lto_pass_manager(
        cgcx: &CodegenContext<Self>,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        thin: bool,
    ) {
        back::lto::run_pass_manager(cgcx, module, config, thin)
    }
}

unsafe impl Send for LlvmCodegenBackend {} // Llvm is on a per-thread basis
unsafe impl Sync for LlvmCodegenBackend {}

impl LlvmCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(LlvmCodegenBackend(()))
    }
}

impl CodegenBackend for LlvmCodegenBackend {
    fn init(&self, sess: &Session) {
        llvm_util::init(sess); // Make sure llvm is inited
    }

    fn print(&self, req: PrintRequest, sess: &Session) {
        match req {
            PrintRequest::RelocationModels => {
                println!("Available relocation models:");
                for name in
                    &["static", "pic", "dynamic-no-pic", "ropi", "rwpi", "ropi-rwpi", "default"]
                {
                    println!("    {}", name);
                }
                println!();
            }
            PrintRequest::CodeModels => {
                println!("Available code models:");
                for name in &["tiny", "small", "kernel", "medium", "large"] {
                    println!("    {}", name);
                }
                println!();
            }
            PrintRequest::TlsModels => {
                println!("Available TLS models:");
                for name in &["global-dynamic", "local-dynamic", "initial-exec", "local-exec"] {
                    println!("    {}", name);
                }
                println!();
            }
            req => llvm_util::print(req, sess),
        }
    }

    fn print_passes(&self) {
        llvm_util::print_passes();
    }

    fn print_version(&self) {
        llvm_util::print_version();
    }

    fn target_features(&self, sess: &Session) -> Vec<Symbol> {
        target_features(sess)
    }

    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(metadata::LlvmMetadataLoader)
    }

    fn provide(&self, providers: &mut ty::query::Providers) {
        attributes::provide_both(providers);
    }

    fn provide_extern(&self, providers: &mut ty::query::Providers) {
        attributes::provide_both(providers);
    }

    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            LlvmCodegenBackend(()),
            tcx,
            metadata,
            need_metadata_module,
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
    ) -> Result<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>), ErrorReported> {
        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<LlvmCodegenBackend>>()
            .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess);

        sess.time("llvm_dump_timing_file", || {
            if sess.opts.debugging_opts.llvm_time_trace {
                llvm_util::time_trace_profiler_finish("llvm_timings.json");
            }
        });

        Ok((codegen_results, work_products))
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported> {
        use crate::back::archive::LlvmArchiveBuilder;
        use rustc_codegen_ssa::back::link::link_binary;

        // Run the linker on any artifacts that resulted from the LLVM run.
        // This should produce either a finished executable or library.
        let target_cpu = crate::llvm_util::target_cpu(sess);
        link_binary::<LlvmArchiveBuilder<'_>>(
            sess,
            &codegen_results,
            outputs,
            &codegen_results.crate_name.as_str(),
            target_cpu,
        );

        Ok(())
    }
}

pub struct ModuleLlvm {
    llcx: &'static mut llvm::Context,
    llmod_raw: *const llvm::Module,
    tm: &'static mut llvm::TargetMachine,
}

unsafe impl Send for ModuleLlvm {}
unsafe impl Sync for ModuleLlvm {}

impl ModuleLlvm {
    fn new(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm { llmod_raw, llcx, tm: create_target_machine(tcx, mod_name) }
        }
    }

    fn new_metadata(tcx: TyCtxt<'_>, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;
            ModuleLlvm { llmod_raw, llcx, tm: create_informational_target_machine(tcx.sess) }
        }
    }

    fn parse(
        cgcx: &CodegenContext<LlvmCodegenBackend>,
        name: &CStr,
        buffer: &[u8],
        handler: &Handler,
    ) -> Result<Self, FatalError> {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
            let llmod_raw = back::lto::parse_module(llcx, name, buffer, handler)?;

            let split_dwarf_file = cgcx
                .output_filenames
                .split_dwarf_filename(cgcx.split_dwarf_kind, Some(name.to_str().unwrap()));
            let tm_factory_config = TargetMachineFactoryConfig { split_dwarf_file };

            let tm = match (cgcx.tm_factory)(tm_factory_config) {
                Ok(m) => m,
                Err(e) => {
                    handler.struct_err(&e).emit();
                    return Err(FatalError);
                }
            };

            Ok(ModuleLlvm { llmod_raw, llcx, tm })
        }
    }

    fn llmod(&self) -> &llvm::Module {
        unsafe { &*self.llmod_raw }
    }
}

impl Drop for ModuleLlvm {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMContextDispose(&mut *(self.llcx as *mut _));
            llvm::LLVMRustDisposeTargetMachine(&mut *(self.tm as *mut _));
        }
    }
}
