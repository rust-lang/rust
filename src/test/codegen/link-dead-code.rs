// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Clink-dead-code

#![feature(const_fn)]
#![crate_type = "rlib"]

// This test makes sure that, when -Clink-dead-code is specified, we generate
// code for functions that would otherwise be skipped.

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code8const_fn
const fn const_fn() -> i32 { 1 }

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code9inline_fn
#[inline]
fn inline_fn() -> i32 { 2 }

// CHECK-LABEL: define hidden i32 @_ZN14link_dead_code10private_fn
fn private_fn() -> i32 { 3 }
