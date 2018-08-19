// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #46314

#![feature(nll)]
#![feature(decl_macro)]

struct NonCopy(String);

struct Wrapper {
    inner: NonCopy,
}

macro inner_copy($wrapper:ident) {
    $wrapper.inner
}

fn main() {
    let wrapper = Wrapper {
        inner: NonCopy("foo".into()),
    };
    assert_two_non_copy(
        inner_copy!(wrapper),
        wrapper.inner,
        //~^ ERROR use of moved value: `wrapper.inner` [E0382]
    );
}

fn assert_two_non_copy(a: NonCopy, b: NonCopy) {
    assert_eq!(a.0, b.0);
}
