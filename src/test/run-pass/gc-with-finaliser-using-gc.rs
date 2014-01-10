// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(experimental)];

// we need to make sure that a finaliser that needs to talk to the GC
// can run (because this happens inside a collection, when the GC is
// borrowed from the task)

use std::libvec::Vec;
use std::gc::Gc;

static mut dtor_actually_ran: bool = false;
struct Foo(uint);
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {dtor_actually_ran = true;}
    }
}

// put some more data on to the stack, so that the Gc::new() pointer
// below doesn't get picked up by the conservative stack scan of the
// collections in the for loop below.

#[inline(never)]
fn make_some_stack_frames(n: uint) {
    if n == 0 {
        let mut v = Vec::new();
        let p  = Gc::new(Foo(1));
        v.push(p);
        Gc::new(v);
    } else {
        make_some_stack_frames(n - 1);
    }
}

fn main() {
    make_some_stack_frames(100);

    for _ in range(0, 10000) {
        Gc::new(10);
    }

    assert!(unsafe {dtor_actually_ran});
}
