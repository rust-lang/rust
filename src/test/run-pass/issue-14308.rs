// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A(int);
struct B;

fn main() {
    let x = match A(3) {
        A(..) => 1
    };
    assert_eq!(x, 1);
    let x = match A(4) {
        A(1) => 1,
        A(..) => 2
    };
    assert_eq!(x, 2);

    // This next test uses a (..) wildcard match on a nullary struct.
    // There's no particularly good reason to support this, but it's currently allowed,
    // and this makes sure it doesn't ICE or break LLVM.
    let x = match B {
        B(..) => 3
    };
    assert_eq!(x, 3);
}
