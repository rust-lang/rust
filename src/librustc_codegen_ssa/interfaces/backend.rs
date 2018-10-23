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
use super::write::WriteBackendMethods;
use rustc::session::Session;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc::middle::cstore::EncodedMetadata;
use rustc::middle::allocator::AllocatorKind;
use rustc::ty::TyCtxt;
use rustc::mir::mono::Stats;
use syntax_pos::symbol::InternedString;
use std::sync::Arc;

pub trait Backend<'ll> {
    type Value : 'll + CodegenObject;
    type BasicBlock : Copy;
    type Type : CodegenObject;
    type Context;
}

pub trait ExtraBackendMethods : CodegenBackend + WriteBackendMethods + Sized + Send + Sync {
    fn thin_lto_available(&self) -> bool;
    fn pgo_available(&self) -> bool;
    fn new_metadata(&self, sess: &Session, mod_name: &str) -> Self::Module;
    fn write_metadata<'b, 'gcx>(
        &self,
        tcx: TyCtxt<'b, 'gcx, 'gcx>,
        metadata: &Self::Module
    ) -> EncodedMetadata;
    fn codegen_allocator(&self, tcx: TyCtxt, mods: &Self::Module, kind: AllocatorKind);
    fn compile_codegen_unit<'ll, 'tcx: 'll>(
        &self,
        tcx: TyCtxt<'ll, 'tcx, 'tcx>,
        cgu_name: InternedString
    ) -> Stats ;
    // If find_features is true this won't access `sess.crate_types` by assuming
    // that `is_pie_binary` is false. When we discover LLVM target features
    // `sess.crate_types` is uninitialized so we cannot access it.
    fn target_machine_factory(
        &self,
        sess: &Session,
        find_features: bool
    ) -> Arc<dyn Fn() ->
        Result<Self::TargetMachine, String> + Send + Sync>;
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str;
}
