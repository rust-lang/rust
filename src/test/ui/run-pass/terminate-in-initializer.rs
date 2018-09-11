// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

// Issue #787
// Don't try to clean up uninitialized locals


use std::thread;

fn test_break() { loop { let _x: Box<isize> = break; } }

fn test_cont() { let mut i = 0; while i < 1 { i += 1; let _x: Box<isize> = continue; } }

fn test_ret() { let _x: Box<isize> = return; }

fn test_panic() {
    fn f() { let _x: Box<isize> = panic!(); }
    thread::spawn(move|| f() ).join().unwrap_err();
}

fn test_panic_indirect() {
    fn f() -> ! { panic!(); }
    fn g() { let _x: Box<isize> = f(); }
    thread::spawn(move|| g() ).join().unwrap_err();
}

pub fn main() {
    test_break();
    test_cont();
    test_ret();
    test_panic();
    test_panic_indirect();
}
