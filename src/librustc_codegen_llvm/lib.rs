//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(custom_attribute)]
#![feature(extern_types)]
#![feature(in_band_lifetimes)]
#![allow(unused_attributes)]
#![feature(libc)]
#![feature(nll)]
#![feature(range_contains)]
#![feature(rustc_diagnostic_macros)]
#![feature(optin_builtin_traits)]
#![feature(concat_idents)]
#![feature(link_args)]
#![feature(static_nobundle)]
#![deny(rust_2018_idioms)]
#![allow(explicit_outlives_requirements)]
#![allow(elided_lifetimes_in_paths)]

use back::write::create_target_machine;
use syntax_pos::symbol::Symbol;

extern crate flate2;
#[macro_use] extern crate bitflags;
extern crate libc;
#[macro_use] extern crate rustc;
extern crate rustc_mir;
extern crate rustc_allocator;
extern crate rustc_target;
#[macro_use] extern crate rustc_data_structures;
extern crate rustc_incremental;
extern crate rustc_codegen_utils;
extern crate rustc_codegen_ssa;
extern crate rustc_fs_util;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;
extern crate serialize;
extern crate tempfile;

use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::back::write::{CodegenContext, ModuleConfig, FatLTOInput};
use rustc_codegen_ssa::back::lto::{SerializedModule, LtoModuleCodegen, ThinModule};
use rustc_codegen_ssa::CompiledModule;
use errors::{FatalError, Handler};
use rustc::dep_graph::WorkProduct;
use rustc::util::time_graph::Timeline;
use syntax_pos::symbol::InternedString;
use rustc::mir::mono::Stats;
pub use llvm_util::target_features;
use std::any::Any;
use std::sync::{mpsc, Arc};

use rustc::dep_graph::DepGraph;
use rustc::middle::allocator::AllocatorKind;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc::session::{Session, CompileIncomplete};
use rustc::session::config::{OutputFilenames, OutputType, PrintRequest, OptLevel};
use rustc::ty::{self, TyCtxt};
use rustc::util::time_graph;
use rustc::util::profiling::ProfileCategory;
use rustc_mir::monomorphize;
use rustc_codegen_ssa::ModuleCodegen;
use rustc_codegen_utils::codegen_backend::CodegenBackend;

mod diagnostics;

mod back {
    mod archive;
    pub mod bytecode;
    pub mod link;
    pub mod lto;
    pub mod write;
    mod rpath;
    pub mod wasm;
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
mod debuginfo;
mod declare;
mod intrinsic;

// The following is a work around that replaces `pub mod llvm;` and that fixes issue 53912.
#[path = "llvm/mod.rs"] mod llvm_; pub mod llvm { pub use super::llvm_::*; }

mod llvm_util;
mod metadata;
mod mono_item;
mod type_;
mod type_of;
mod value;
mod va_arg;

#[derive(Clone)]
pub struct LlvmCodegenBackend(());

impl ExtraBackendMethods for LlvmCodegenBackend {
    fn new_metadata(&self, tcx: TyCtxt, mod_name: &str) -> ModuleLlvm {
        ModuleLlvm::new(tcx, mod_name)
    }
    fn write_metadata<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'gcx>,
        metadata: &ModuleLlvm
    ) -> EncodedMetadata {
        base::write_metadata(tcx, metadata)
    }
    fn codegen_allocator(&self, tcx: TyCtxt, mods: &ModuleLlvm, kind: AllocatorKind) {
        unsafe { allocator::codegen(tcx, mods, kind) }
    }
    fn compile_codegen_unit<'a, 'tcx: 'a>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        cgu_name: InternedString,
    ) -> Stats {
        base::compile_codegen_unit(tcx, cgu_name)
    }
    fn target_machine_factory(
        &self,
        sess: &Session,
        optlvl: OptLevel,
        find_features: bool
    ) -> Arc<dyn Fn() ->
        Result<&'static mut llvm::TargetMachine, String> + Send + Sync> {
        back::write::target_machine_factory(sess, optlvl, find_features)
    }
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str {
        llvm_util::target_cpu(sess)
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
            unsafe { llvm::LLVMRustPrintPassTimings(); }
    }
    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLTOInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
        timeline: &mut Timeline
    ) -> Result<LtoModuleCodegen<Self>, FatalError> {
        back::lto::run_fat(cgcx, modules, cached_modules, timeline)
    }
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
        timeline: &mut Timeline
    ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        back::lto::run_thin(cgcx, modules, cached_modules, timeline)
    }
    unsafe fn optimize(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        timeline: &mut Timeline
    ) -> Result<(), FatalError> {
        back::write::optimize(cgcx, diag_handler, module, config, timeline)
    }
    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: &mut ThinModule<Self>,
        timeline: &mut Timeline
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        back::lto::optimize_thin_module(thin, cgcx, timeline)
    }
    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        diag_handler: &Handler,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        timeline: &mut Timeline
    ) -> Result<CompiledModule, FatalError> {
        back::write::codegen(cgcx, diag_handler, module, config, timeline)
    }
    fn prepare_thin(
        module: ModuleCodegen<Self::Module>
    ) -> (String, Self::ThinBuffer) {
        back::lto::prepare_thin(module)
    }
    fn serialize_module(
        module: ModuleCodegen<Self::Module>
    ) -> (String, Self::ModuleBuffer) {
        (module.name, back::lto::ModuleBuffer::new(module.module_llvm.llmod()))
    }
    fn run_lto_pass_manager(
        cgcx: &CodegenContext<Self>,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
        thin: bool
    ) {
        back::lto::run_pass_manager(cgcx, module, config, thin)
    }
}

unsafe impl Send for LlvmCodegenBackend {} // Llvm is on a per-thread basis
unsafe impl Sync for LlvmCodegenBackend {}

impl LlvmCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        box LlvmCodegenBackend(())
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
                for &(name, _) in back::write::RELOC_MODEL_ARGS.iter() {
                    println!("    {}", name);
                }
                println!("");
            }
            PrintRequest::CodeModels => {
                println!("Available code models:");
                for &(name, _) in back::write::CODE_GEN_MODEL_ARGS.iter(){
                    println!("    {}", name);
                }
                println!("");
            }
            PrintRequest::TlsModels => {
                println!("Available TLS models:");
                for &(name, _) in back::write::TLS_MODEL_ARGS.iter(){
                    println!("    {}", name);
                }
                println!("");
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

    fn diagnostics(&self) -> &[(&'static str, &'static str)] {
        &DIAGNOSTICS
    }

    fn target_features(&self, sess: &Session) -> Vec<Symbol> {
        target_features(sess)
    }

    fn metadata_loader(&self) -> Box<dyn MetadataLoader + Sync> {
        box metadata::LlvmMetadataLoader
    }

    fn provide(&self, providers: &mut ty::query::Providers) {
        rustc_codegen_utils::symbol_names::provide(providers);
        rustc_codegen_ssa::back::symbol_export::provide(providers);
        rustc_codegen_ssa::base::provide_both(providers);
        attributes::provide(providers);
    }

    fn provide_extern(&self, providers: &mut ty::query::Providers) {
        rustc_codegen_ssa::back::symbol_export::provide_extern(providers);
        rustc_codegen_ssa::base::provide_both(providers);
        attributes::provide_extern(providers);
    }

    fn codegen_crate<'b, 'tcx>(
        &self,
        tcx: TyCtxt<'b, 'tcx, 'tcx>,
        rx: mpsc::Receiver<Box<dyn Any + Send>>
    ) -> Box<dyn Any> {
        box rustc_codegen_ssa::base::codegen_crate(LlvmCodegenBackend(()), tcx, rx)
    }

    fn join_codegen_and_link(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        dep_graph: &DepGraph,
        outputs: &OutputFilenames,
    ) -> Result<(), CompileIncomplete>{
        use rustc::util::common::time;
        let (codegen_results, work_products) =
            ongoing_codegen.downcast::
                <rustc_codegen_ssa::back::write::OngoingCodegen<LlvmCodegenBackend>>()
                .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
                .join(sess);
        if sess.opts.debugging_opts.incremental_info {
            rustc_codegen_ssa::back::write::dump_incremental_data(&codegen_results);
        }

        time(sess,
             "serialize work products",
             move || rustc_incremental::save_work_product_index(sess, &dep_graph, work_products));

        sess.compile_status()?;

        if !sess.opts.output_types.keys().any(|&i| i == OutputType::Exe ||
                                                   i == OutputType::Metadata) {
            return Ok(());
        }

        // Run the linker on any artifacts that resulted from the LLVM run.
        // This should produce either a finished executable or library.
        sess.profiler(|p| p.start_activity(ProfileCategory::Linking));
        time(sess, "linking", || {
            back::link::link_binary(sess, &codegen_results,
                                    outputs, &codegen_results.crate_name.as_str());
        });
        sess.profiler(|p| p.end_activity(ProfileCategory::Linking));

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(sess, codegen_results.crate_hash);

        Ok(())
    }
}

/// This is the entrypoint for a hot plugged rustc_codegen_llvm
#[no_mangle]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    LlvmCodegenBackend::new()
}

pub struct ModuleLlvm {
    llcx: &'static mut llvm::Context,
    llmod_raw: *const llvm::Module,
    tm: &'static mut llvm::TargetMachine,
}

unsafe impl Send for ModuleLlvm { }
unsafe impl Sync for ModuleLlvm { }

impl ModuleLlvm {
    fn new(tcx: TyCtxt, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(tcx.sess.fewer_names());
            let llmod_raw = context::create_module(tcx, llcx, mod_name) as *const _;

            ModuleLlvm {
                llmod_raw,
                llcx,
                tm: create_target_machine(tcx, false),
            }
        }
    }

    fn parse(
        cgcx: &CodegenContext<LlvmCodegenBackend>,
        name: &str,
        buffer: &back::lto::ModuleBuffer,
        handler: &Handler,
    ) -> Result<Self, FatalError> {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(cgcx.fewer_names);
            let llmod_raw = buffer.parse(name, llcx, handler)?;
            let tm = match (cgcx.tm_factory.0)() {
                Ok(m) => m,
                Err(e) => {
                    handler.struct_err(&e).emit();
                    return Err(FatalError)
                }
            };

            Ok(ModuleLlvm {
                llmod_raw,
                llcx,
                tm,
            })
        }
    }

    fn llmod(&self) -> &llvm::Module {
        unsafe {
            &*self.llmod_raw
        }
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

__build_diagnostic_array! { librustc_codegen_llvm, DIAGNOSTICS }
