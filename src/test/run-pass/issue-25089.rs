// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

use std::thread;

struct Foo(i32);

impl Drop for Foo {
    fn drop(&mut self) {
        static mut DROPPED: bool = false;
        unsafe {
            assert!(!DROPPED);
            DROPPED = true;
        }
    }
}

struct Empty;

fn empty() -> Empty { Empty }

fn should_panic(_: Foo, _: Empty) {
    panic!("test panic");
}

fn test() {
    should_panic(Foo(1), empty());
}

fn main() {
    let ret = thread::spawn(test).join();
    assert!(ret.is_err());
}
