// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::CodegenObject;
use {ModuleCodegen, CachedModuleCodegen};
use rustc::session::Session;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc::middle::cstore::EncodedMetadata;
use rustc::middle::allocator::AllocatorKind;
use rustc::ty::TyCtxt;
use rustc::mir::mono::Stats;
use syntax_pos::symbol::InternedString;
use rustc::util::time_graph::TimeGraph;
use std::sync::mpsc::Receiver;
use std::any::Any;

pub trait Backend<'ll> {
    type Value : 'll + CodegenObject;
    type BasicBlock : Copy;
    type Type : CodegenObject;
    type Context;
}

pub trait ExtraBackendMethods : CodegenBackend {
    type Metadata;
    type OngoingCodegen;

    fn thin_lto_available(&self) -> bool;
    fn pgo_available(&self) -> bool;
    fn new_metadata(&self, sess: &Session, mod_name: &str) -> Self::Metadata;
    fn write_metadata<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'gcx>,
        metadata: &Self::Metadata
    ) -> EncodedMetadata;
    fn codegen_allocator(&self, tcx: TyCtxt, mods: &Self::Metadata, kind: AllocatorKind);

    fn start_async_codegen(
        &self,
        tcx: TyCtxt,
        time_graph: Option<TimeGraph>,
        metadata: EncodedMetadata,
        coordinator_receive: Receiver<Box<dyn Any + Send>>,
        total_cgus: usize
    ) -> Self::OngoingCodegen;
    fn submit_pre_codegened_module_to_llvm(
        &self,
        codegen: &Self::OngoingCodegen,
        tcx: TyCtxt,
        module: ModuleCodegen<Self::Metadata>
    );
    fn submit_pre_lto_module_to_llvm(&self, tcx: TyCtxt, module: CachedModuleCodegen);
    fn submit_post_lto_module_to_llvm(&self, tcx: TyCtxt, module: CachedModuleCodegen);
    fn codegen_finished(&self, codegen: &Self::OngoingCodegen, tcx: TyCtxt);
    fn check_for_errors(&self, codegen: &Self::OngoingCodegen, sess: &Session);
    fn wait_for_signal_to_codegen_item(&self, codegen: &Self::OngoingCodegen);
    fn compile_codegen_unit<'ll, 'tcx: 'll>(
        &self,
        tcx: TyCtxt<'ll, 'tcx, 'tcx>,
        cgu_name: InternedString
    ) -> Stats ;
}
