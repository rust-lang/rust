// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::thread;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("test2");
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    // Off the main thread due to #28129, be sure to initialize FOO first before
    // calling `println!`
    thread::spawn(|| {
        FOO.with(|_| {});
        println!("test1");
    }).join().unwrap();
}
