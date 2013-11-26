// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// Issue #787
// Don't try to clean up uninitialized locals

extern mod extra;

use std::task;

fn test_break() { loop { let _x: @int = break; } }

fn test_cont() { let mut i = 0; while i < 1 { i += 1; let _x: @int = continue; } }

fn test_ret() { let _x: @int = return; }

fn test_fail() {
    fn f() { let _x: @int = fail!(); }
    task::try(proc() f() );
}

fn test_fail_indirect() {
    fn f() -> ! { fail!(); }
    fn g() { let _x: @int = f(); }
    task::try(proc() g() );
}

pub fn main() {
    test_break();
    test_cont();
    test_ret();
    test_fail();
    test_fail_indirect();
}
