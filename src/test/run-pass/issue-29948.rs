// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-wasm32-bare compiled with panic=abort by default

use std::panic;

impl<'a> panic::UnwindSafe for Foo<'a> {}
impl<'a> panic::RefUnwindSafe for Foo<'a> {}

struct Foo<'a>(&'a mut bool);

impl<'a> Drop for Foo<'a> {
    fn drop(&mut self) {
        *self.0 = true;
    }
}

fn f<T: FnOnce()>(t: T) {
    t()
}

fn main() {
    let mut ran_drop = false;
    {
        let x = Foo(&mut ran_drop);
        let x = move || { let _ = x; };
        f(x);
    }
    assert!(ran_drop);

    let mut ran_drop = false;
    {
        let x = Foo(&mut ran_drop);
        let result = panic::catch_unwind(move || {
            let x = move || { let _ = x; panic!() };
            f(x);
        });
        assert!(result.is_err());
    }
    assert!(ran_drop);
}
