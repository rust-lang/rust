// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the trace_macros feature gate is on.

pub fn main() {
    println!("arg: {}", trace_macros!()); //~ ERROR `trace_macros` is not stable
    println!("arg: {}", trace_macros!(1)); //~ ERROR `trace_macros` is not stable
    println!("arg: {}", trace_macros!(ident)); //~ ERROR `trace_macros` is not stable
    println!("arg: {}", trace_macros!(for)); //~ ERROR `trace_macros` is not stable
    println!("arg: {}", trace_macros!(true,)); //~ ERROR `trace_macros` is not stable
    println!("arg: {}", trace_macros!(false 1)); //~ ERROR `trace_macros` is not stable
}
