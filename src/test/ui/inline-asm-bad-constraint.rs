// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the compiler will catch invalid inline assembly constraints.

// ignore-emscripten

#![feature(asm)]

extern "C" {
    fn foo(a: usize);
}

fn main() {
    bad_register_constraint();
    bad_input();
    wrong_size_output();
}

// Issue #54130
fn bad_register_constraint() {
    let rax: u64;
    unsafe {
        asm!("" :"={rax"(rax)) //~ ERROR E0668
    };
    println!("Accumulator is: {}", rax);
}

// Issue #54376
fn bad_input() {
    unsafe {
        asm!("callq $0" : : "0"(foo)) //~ ERROR E0668
    };
}

fn wrong_size_output() {
    let rax: u64 = 0;
    unsafe {
        asm!("addb $1, $0" : "={rax}"((0i32, rax))); //~ ERROR E0668
    }
    println!("rax: {}", rax);
}
