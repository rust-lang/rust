// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:test-macros.rs
// ignore-stage1
// ignore-wasm32

#![feature(proc_macro)]

extern crate test_macros;

use test_macros::{nop_attr, no_output, emit_input};

fn main() {
    assert_eq!(unsafe { rust_get_test_int() }, 0isize);
    assert_eq!(unsafe { rust_dbg_extern_identity_u32(0xDEADBEEF) }, 0xDEADBEEF);
}

#[link(name = "rust_test_helpers", kind = "static")]
extern {
    #[no_output]
    //~^ ERROR Macro and proc-macro invocations in `extern {}` blocks are experimental.
    fn some_definitely_unknown_symbol_which_should_be_removed();

    #[nop_attr]
    //~^ ERROR Macro and proc-macro invocations in `extern {}` blocks are experimental.
    fn rust_get_test_int() -> isize;

    emit_input!(fn rust_dbg_extern_identity_u32(arg: u32) -> u32;);
    //~^ ERROR Macro and proc-macro invocations in `extern {}` blocks are experimental.
}
