// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(crate_visibility_modifier)]
#![feature(custom_attribute)]
#![feature(extern_types)]
#![feature(in_band_lifetimes)]
#![allow(unused_attributes)]
#![feature(libc)]
#![feature(nll)]
#![feature(quote)]
#![feature(range_contains)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_sort_by_cached_key)]
#![feature(optin_builtin_traits)]
#![feature(concat_idents)]
#![feature(link_args)]
#![feature(static_nobundle)]

use back::write::create_target_machine;
use syntax_pos::symbol::Symbol;

extern crate flate2;
#[macro_use] extern crate bitflags;
extern crate libc;
#[macro_use] extern crate rustc;
extern crate jobserver;
extern crate num_cpus;
extern crate rustc_mir;
extern crate rustc_allocator;
extern crate rustc_apfloat;
extern crate rustc_target;
#[macro_use] extern crate rustc_data_structures;
extern crate rustc_demangle;
extern crate rustc_incremental;
extern crate rustc_llvm;
extern crate rustc_platform_intrinsics as intrinsics;
extern crate rustc_codegen_utils;
extern crate rustc_codegen_ssa;
extern crate rustc_fs_util;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;
extern crate serialize;
extern crate cc; // Used to locate MSVC
extern crate tempfile;
extern crate memmap;

use rustc_codegen_ssa::interfaces::*;
use time_graph::TimeGraph;
use std::sync::mpsc::Receiver;
use back::write::{self, OngoingCodegen};
use syntax_pos::symbol::InternedString;
use rustc::mir::mono::Stats;

pub use llvm_util::target_features;
use std::any::Any;
use std::sync::mpsc;

use rustc::dep_graph::DepGraph;
use rustc::middle::allocator::AllocatorKind;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoader};
use rustc::session::{Session, CompileIncomplete};
use rustc::session::config::{OutputFilenames, OutputType, PrintRequest};
use rustc::ty::{self, TyCtxt};
use rustc::util::time_graph;
use rustc::util::profiling::ProfileCategory;
use rustc_mir::monomorphize;
use rustc_codegen_ssa::{ModuleCodegen, CompiledModule, CachedModuleCodegen, CrateInfo};
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_data_structures::svh::Svh;

mod diagnostics;

mod back {
    pub use rustc_codegen_utils::symbol_names;
    mod archive;
    pub mod bytecode;
    mod command;
    pub mod linker;
    pub mod link;
    pub mod lto;
    pub mod symbol_export;
    pub mod write;
    mod rpath;
    pub mod wasm;
}

mod abi;
mod allocator;
mod asm;
mod attributes;
mod base;
mod callee;
mod builder;
mod common;
mod consts;
mod context;
mod debuginfo;
mod declare;
mod intrinsic;
mod mono_item;

// The following is a work around that replaces `pub mod llvm;` and that fixes issue 53912.
#[path = "llvm/mod.rs"] mod llvm_; pub mod llvm { pub use super::llvm_::*; }

mod llvm_util;
mod metadata;
mod type_;
mod type_of;
mod value;

pub struct LlvmCodegenBackend(());

impl ExtraBackendMethods for LlvmCodegenBackend {
    type Metadata = ModuleLlvm;
    type OngoingCodegen = OngoingCodegen;

    fn thin_lto_available(&self) -> bool {
         unsafe { !llvm::LLVMRustThinLTOAvailable() }
    }
    fn pgo_available(&self) -> bool {
        unsafe { !llvm::LLVMRustPGOAvailable() }
    }
    fn new_metadata(&self, sess: &Session, mod_name: &str) -> ModuleLlvm {
        ModuleLlvm::new(sess, mod_name)
    }
    fn write_metadata<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'gcx>,
        metadata: &ModuleLlvm
    ) -> EncodedMetadata {
        base::write_metadata(tcx, metadata)
    }
    fn start_async_codegen(
        &self,
        tcx: TyCtxt,
        time_graph: Option<TimeGraph>,
        metadata: EncodedMetadata,
        coordinator_receive: Receiver<Box<dyn Any + Send>>,
        total_cgus: usize
    ) -> OngoingCodegen {
        write::start_async_codegen(tcx, time_graph, metadata, coordinator_receive, total_cgus)
    }
    fn submit_pre_codegened_module_to_llvm(
        &self,
        codegen : &OngoingCodegen,
        tcx: TyCtxt,
        module: ModuleCodegen<ModuleLlvm>
    ) {
        codegen.submit_pre_codegened_module_to_llvm(tcx, module)
    }
    fn submit_pre_lto_module_to_llvm(&self, tcx: TyCtxt, module: CachedModuleCodegen) {
        write::submit_pre_lto_module_to_llvm(tcx, module)
    }
    fn submit_post_lto_module_to_llvm(&self, tcx: TyCtxt, module: CachedModuleCodegen) {
        write::submit_post_lto_module_to_llvm(tcx, module)
    }
    fn codegen_finished(&self, codegen : &OngoingCodegen, tcx: TyCtxt) {
        codegen.codegen_finished(tcx)
    }
    fn check_for_errors(&self, codegen: &OngoingCodegen, sess: &Session) {
        codegen.check_for_errors(sess)
    }
    fn codegen_allocator(&self, tcx: TyCtxt, mods: &ModuleLlvm, kind: AllocatorKind) {
        unsafe { allocator::codegen(tcx, mods, kind) }
    }
    fn wait_for_signal_to_codegen_item(&self, codegen: &OngoingCodegen) {
        codegen.wait_for_signal_to_codegen_item()
    }
    fn compile_codegen_unit<'ll, 'tcx: 'll>(
        &self,
        tcx: TyCtxt<'ll, 'tcx, 'tcx>,
        cgu_name: InternedString
    ) -> Stats {
        base::compile_codegen_unit(tcx, cgu_name)
    }
}


impl !Send for LlvmCodegenBackend {} // Llvm is on a per-thread basis
impl !Sync for LlvmCodegenBackend {}

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
        back::symbol_names::provide(providers);
        back::symbol_export::provide(providers);
        rustc_codegen_ssa::base::provide(providers);
        attributes::provide(providers);
    }

    fn provide_extern(&self, providers: &mut ty::query::Providers) {
        back::symbol_export::provide_extern(providers);
        rustc_codegen_ssa::base::provide_extern(providers);
        attributes::provide_extern(providers);
    }

    fn codegen_crate<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
        let (ongoing_codegen, work_products) =
            ongoing_codegen.downcast::<::back::write::OngoingCodegen>()
                .expect("Expected LlvmCodegenBackend's OngoingCodegen, found Box<Any>")
                .join(sess);
        if sess.opts.debugging_opts.incremental_info {
            back::write::dump_incremental_data(&ongoing_codegen);
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
            back::link::link_binary(sess, &ongoing_codegen,
                                    outputs, &ongoing_codegen.crate_name.as_str());
        });
        sess.profiler(|p| p.end_activity(ProfileCategory::Linking));

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(sess, ongoing_codegen.crate_hash);

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
    fn new(sess: &Session, mod_name: &str) -> Self {
        unsafe {
            let llcx = llvm::LLVMRustContextCreate(sess.fewer_names());
            let llmod_raw = context::create_module(sess, llcx, mod_name) as *const _;

            ModuleLlvm {
                llmod_raw,
                llcx,
                tm: create_target_machine(sess, false),
            }
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

struct CodegenResults {
    crate_name: Symbol,
    modules: Vec<CompiledModule>,
    allocator_module: Option<CompiledModule>,
    metadata_module: CompiledModule,
    crate_hash: Svh,
    metadata: rustc::middle::cstore::EncodedMetadata,
    windows_subsystem: Option<String>,
    linker_info: back::linker::LinkerInfo,
    crate_info: CrateInfo,
}
__build_diagnostic_array! { librustc_codegen_llvm, DIAGNOSTICS }
