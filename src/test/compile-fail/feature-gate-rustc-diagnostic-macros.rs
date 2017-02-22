// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that diagnostic macros are gated by `rustc_diagnostic_macros` feature
// gate

__register_diagnostic!(E0001);
//~^ ERROR cannot find macro `__register_diagnostic!` in this scope

fn main() {
    __diagnostic_used!(E0001);
    //~^ ERROR cannot find macro `__diagnostic_used!` in this scope
}

__build_diagnostic_array!(DIAGNOSTICS);
//~^ ERROR cannot find macro `__build_diagnostic_array!` in this scope
