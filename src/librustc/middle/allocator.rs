// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Clone, Copy)]
pub enum AllocatorKind {
    Global,
    Default,
}

impl AllocatorKind {
    pub fn fn_name(&self, base: &str) -> String {
        match *self {
            AllocatorKind::Global => format!("__rg_{}", base),
            AllocatorKind::Default => format!("__rdl_{}", base),
        }
    }
}
