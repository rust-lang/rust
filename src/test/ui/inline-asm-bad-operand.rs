// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the compiler will catch passing invalid values to inline assembly
// operands.

// ignore-emscripten

#![feature(asm)]

#[repr(C)]
struct MyPtr(usize);

fn main() {
    issue_37433();
    issue_37437();
    issue_40187();
    issue_54067();
    multiple_errors();
}

fn issue_37433() {
    unsafe {
        asm!("" :: "r"("")); //~ ERROR E0669
    }

    unsafe {
        let target = MyPtr(0);
        asm!("ret" : : "{rdi}"(target)); //~ ERROR E0669
    }
}

fn issue_37437() {
    let hello: &str = "hello";
    // this should fail...
    unsafe { asm!("" :: "i"(hello)) }; //~ ERROR E0669
    // but this should succeed.
    unsafe { asm!("" :: "r"(hello.as_ptr())) };
}

fn issue_40187() {
    let arr: [u8; 1] = [0; 1];
    unsafe {
        asm!("movups $1, %xmm0"::"m"(arr)); //~ ERROR E0669
    }
}

fn issue_54067() {
    let addr: Option<u32> = Some(123);
    unsafe {
        asm!("mov sp, $0"::"r"(addr)); //~ ERROR E0669
    }
}

fn multiple_errors() {
    let addr: (u32, u32) = (1, 2);
    unsafe {
        asm!("mov sp, $0"::"r"(addr), //~ ERROR E0669
                           "r"("hello e0669")); //~ ERROR E0669
    }
}
