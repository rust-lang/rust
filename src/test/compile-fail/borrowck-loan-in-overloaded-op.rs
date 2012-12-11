// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test #3387

enum foo = ~uint;

impl foo : Add<foo, foo> {
    pure fn add(f: &foo) -> foo {
        foo(~(**self + **(*f)))
    }
}

fn main() {
    let x = foo(~3);
    let _y = x + move x;
    //~^ ERROR moving out of immutable local variable prohibited due to outstanding loan
}
