// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

thread_local!(static FOO: Foo = Foo);
thread_local!(static BAR: Bar = Bar(1));
thread_local!(static BAZ: Baz = Baz);

static mut HIT: bool = false;

struct Foo;
struct Bar(i32);
struct Baz;

impl Drop for Foo {
    fn drop(&mut self) {
        BAR.with(|_| {});
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        assert_eq!(self.0, 1);
        self.0 = 2;
        BAZ.with(|_| {});
        assert_eq!(self.0, 2);
    }
}

impl Drop for Baz {
    fn drop(&mut self) {
        unsafe { HIT = true; }
    }
}

fn main() {
    std::thread::spawn(|| {
        FOO.with(|_| {});
    }).join().unwrap();
    assert!(unsafe { HIT });
}
