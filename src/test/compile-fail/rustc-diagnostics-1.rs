// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_diagnostic_macros)]

__register_diagnostic!(E0001)
__register_diagnostic!(E0003)

fn main() {
    __diagnostic_used!(E0002);
    //~^ ERROR unknown diagnostic code E0002

    __diagnostic_used!(E0001);
    //~^ NOTE previous invocation

    __diagnostic_used!(E0001);
    //~^ WARNING diagnostic code E0001 already used
}

__build_diagnostic_array!(DIAGNOSTICS)
//~^ WARN diagnostic code E0003 never used
