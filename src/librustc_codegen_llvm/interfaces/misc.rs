// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::backend::Backend;
use rustc::ty::{self, Instance, Ty};
use rustc::util::nodemap::FxHashMap;
use std::cell::RefCell;

pub trait MiscMethods<'tcx>: Backend<'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, ty::PolyExistentialTraitRef<'tcx>), Self::Value>>;
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Value;
}
