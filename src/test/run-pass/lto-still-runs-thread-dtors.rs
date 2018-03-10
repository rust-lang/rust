// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C lto
// no-prefer-dynamic
// ignore-emscripten no threads support

use std::thread;

static mut HIT: usize = 0;

thread_local!(static A: Foo = Foo);

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            HIT += 1;
        }
    }
}

fn main() {
    unsafe {
        assert_eq!(HIT, 0);
        thread::spawn(|| {
            assert_eq!(HIT, 0);
            A.with(|_| ());
            assert_eq!(HIT, 0);
        }).join().unwrap();
        assert_eq!(HIT, 1);
    }
}
