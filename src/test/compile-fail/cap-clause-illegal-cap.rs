// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: copying a noncopyable value

struct foo { x: int, }

impl foo : Drop {
    fn finalize(&self) {}
}

fn foo(x: int) -> foo {
    foo {
        x: x
    }
}

fn to_lambda2(b: foo) -> fn@(uint) -> uint {
    // test case where copy clause specifies a value that is not used
    // in fn@ body, but value is illegal to copy:
    return fn@(u: uint, copy b) -> uint { 22u };
}

fn main() {
}
