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
// min-lldb-version: 310

// compile-flags:-g

#![allow(unused_variables)]
#![omit_gdb_pretty_printer_section]

// No need to actually run the debugger, just make sure that the compiler can
// handle locals in unreachable code.

fn after_return() {
    return;
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in [111i32].iter() {}
}

fn after_panic() {
    panic!();
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in [111i32].iter() {}
}

fn after_diverging_function() {
    diverge();
    let x = "0";
    let (ref y,z) = (1i32, 2u32);
    match (20i32, 'c') {
        (a, ref b) => {}
    }
    for a in [111i32].iter() {}
}

fn after_break() {
    loop {
        break;
        let x = "0";
        let (ref y,z) = (1i32, 2u32);
        match (20i32, 'c') {
            (a, ref b) => {}
        }
        for a in [111i32].iter() {}
    }
}

fn after_continue() {
    for _ in range(0, 10i32) {
        break;
        let x = "0";
        let (ref y,z) = (1i32, 2u32);
        match (20i32, 'c') {
            (a, ref b) => {}
        }
        for a in [111i32].iter() {}
    }
}

fn main() {
    after_return();
    after_panic();
    after_diverging_function();
    after_break();
    after_continue();
}

fn diverge() -> ! {
    panic!();
}
