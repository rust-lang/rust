// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic sanity check for `push_unsafe!(EXPR)` and
// `pop_unsafe!(EXPR)`: we can call unsafe code when there are a
// positive number of pushes in the stack, or if we are within a
// normal `unsafe` block, but otherwise cannot.

// ignore-pretty because the `push_unsafe!` and `pop_unsafe!` macros
// are not integrated with the pretty-printer.

#![feature(pushpop_unsafe)]

static mut X: i32 = 0;

unsafe fn f() { X += 1; return; }
fn g() { unsafe { X += 1_000; } return; }

fn check_reset_x(x: i32) -> bool {
    #![allow(unused_parens)] // dont you judge my style choices!
    unsafe {
        let ret = (x == X);
        X = 0;
        ret
    }
}

fn main() {
    // double-check test infrastructure
    assert!(check_reset_x(0));
    unsafe { f(); }
    assert!(check_reset_x(1));
    assert!(check_reset_x(0));
    { g(); }
    assert!(check_reset_x(1000));
    assert!(check_reset_x(0));
    unsafe { f(); g(); g(); }
    assert!(check_reset_x(2001));

    push_unsafe!( { f(); pop_unsafe!( g() ) } );
    assert!(check_reset_x(1_001));
    push_unsafe!( { g(); pop_unsafe!( unsafe { f(); f(); } ) } );
    assert!(check_reset_x(1_002));

    unsafe { push_unsafe!( { f(); pop_unsafe!( { f(); f(); } ) } ); }
    assert!(check_reset_x(3));
    push_unsafe!( { f(); push_unsafe!( { pop_unsafe!( { f(); f(); f(); } ) } ); } );
    assert!(check_reset_x(4));
}
