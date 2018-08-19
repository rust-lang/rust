// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that closures implement `Copy`.

fn call<T, F: FnOnce() -> T>(f: F) -> T { f() }

fn main() {
    let a = 5;
    let hello = || {
        println!("Hello {}", a);
        a
    };

    assert_eq!(5, call(hello.clone()));
    assert_eq!(5, call(hello));
    assert_eq!(5, call(hello));
}
