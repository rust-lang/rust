// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that functions get annotated with the proper
// "target-cpu" attribute in LLVM.

// no-prefer-dynamic
// only-msvc
// compile-flags: -Z cross-lang-lto

#![crate_type = "rlib"]

// CHECK-NOT: @{{.*}}__imp_{{.*}}GLOBAL{{.*}} = global i8*

pub static GLOBAL: u32 = 0;
pub static mut GLOBAL2: u32 = 0;
