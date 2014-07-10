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
// ignore-lldb

// compile-flags:-g

// gdb-command:break 'no-debug-attribute.rs':32
// gdb-command:break 'no-debug-attribute.rs':38
// gdb-command:run

// gdb-command:info locals
// gdb-check:No locals.
// gdb-command:continue

// gdb-command:info locals
// gdb-check:abc = 10
// gdb-command:continue

#![allow(unused_variable)]

fn function_with_debuginfo() {
    let abc = 10u;
    return (); // #break
}

#[no_debug]
fn function_without_debuginfo() {
    let abc = -57i32;
    return (); // #break
}

fn main() {
    function_without_debuginfo();
    function_with_debuginfo();
}

