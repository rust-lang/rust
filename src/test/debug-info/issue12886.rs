// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android: FIXME(#10381)

// compile-flags:-g
// debugger:break issue12886.rs:29
// debugger:run
// debugger:next
// check:[...]30[...]s
// debugger:continue

// IF YOU MODIFY THIS FILE, BE CAREFUL TO ADAPT THE LINE NUMBERS IN THE DEBUGGER COMMANDS

// This test makes sure that gdb does not set unwanted breakpoints in inlined functions. If a
// breakpoint existed in unwrap(), then calling `next` would (when stopped at line 27) would stop
// in unwrap() instead of stepping over the function invocation. By making sure that `s` is
// contained in the output, after calling `next` just once, we can be sure that we did not stop in
// unwrap(). (The testing framework doesn't allow for checking that some text is *not* contained in
// the output, which is why we have to make the test in this kind of roundabout way)
fn bar() -> int {
    let s = Some(5).unwrap();
    s
}

fn main() {
    let _ = bar();
}
