// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use rustc::util::nodemap::FxHashMap;
use rustc::ty::{Ty, self, Instance};
use super::Backend;
use rustc::session::Session;
use libc::c_uint;
use rustc::mir::mono::Stats;
use std::sync::Arc;
use rustc_mir::monomorphize::partitioning::CodegenUnit;

pub trait MiscMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    fn vtables(&self) -> &RefCell<FxHashMap<(Ty<'tcx>,
                                Option<ty::PolyExistentialTraitRef<'tcx>>), Self::Value>>;
    fn check_overflow(&self) -> bool;
    fn instances(&self) -> &RefCell<FxHashMap<Instance<'tcx>, Self::Value>>;
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Value;
    fn get_param(&self, llfn: Self::Value, index: c_uint) -> Self::Value;
    fn eh_personality(&self) -> Self::Value;
    fn eh_unwind_resume(&self) -> Self::Value;
    fn sess(&self) -> &Session;
    fn stats(&self) -> &RefCell<Stats>;
    fn consume_stats(self) -> RefCell<Stats>;
    fn codegen_unit(&self) -> &Arc<CodegenUnit<'tcx>>;
    fn statics_to_rauw(&self) -> &RefCell<Vec<(Self::Value, Self::Value)>>;
    fn env_alloca_allowed(&self) -> bool;
    fn used_statics(&self) -> &RefCell<Vec<Self::Value>>;
    fn set_frame_pointer_elimination(&self, llfn: Self::Value);
    fn apply_target_cpu_attr(&self, llfn: Self::Value);
    fn create_used_variable(&self);
}
