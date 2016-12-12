// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: rpass1 rpass2

#![feature(rustc_attrs)]
#![allow(private_no_mangle_fns)]

#![rustc_partition_reused(module="change_symbol_export_status", cfg="rpass2")]
#![rustc_partition_translated(module="change_symbol_export_status-mod1", cfg="rpass2")]


// This test case makes sure that a change in symbol visibility is detected by
// our dependency tracking. We do this by changing a module's visibility to
// `private` in rpass2, causing the contained function to go from `default` to
// `hidden` visibility.
// The function is marked with #[no_mangle] so it is considered for exporting
// even from an executable. Plain Rust functions are only exported from Rust
// libraries, which our test infrastructure does not support.

#[cfg(rpass1)]
pub mod mod1 {
    #[no_mangle]
    pub fn foo() {}
}

#[cfg(rpass2)]
mod mod1 {
    #[no_mangle]
    pub fn foo() {}
}

fn main() {
    mod1::foo();
}
