// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_POISON_ON_FREE=1

// Test that we root `x` even though it is found in immutable memory,
// because it is moved.

#[feature(managed_boxes)];

fn free<T>(x: @T) {}

struct Foo {
    f: @Bar
}

struct Bar {
    g: int
}

fn lend(x: @Foo) -> int {
    let y = &x.f.g;
    free(x); // specifically here, if x is not rooted, it will be freed
    *y
}

pub fn main() {
    assert_eq!(lend(@Foo {f: @Bar {g: 22}}), 22);
}
