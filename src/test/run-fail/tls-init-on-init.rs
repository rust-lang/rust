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

// Can't include entire panic message because it'd exceed tidy's 100-character line length limit.
// error-pattern:cannot access a TLS value while it is being initialized or during or after

#![feature(thread_local_state)]

use std::thread::{self, LocalKeyState};

struct Foo;

thread_local!(static FOO: Foo = Foo::init());

impl Foo {
    fn init() -> Foo {
        FOO.with(|_| {});
        Foo
    }
}

fn main() {
    thread::spawn(|| {
        FOO.with(|_| {});
    }).join().unwrap();
}
