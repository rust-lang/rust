// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsized_locals)]

use std::fmt;

fn gen_foo() -> Box<fmt::Display> {
    Box::new(Box::new("foo"))
}

fn foo(x: fmt::Display) {
    assert_eq!(x.to_string(), "foo");
}

fn foo_indirect(x: fmt::Display) {
    foo(x);
}

fn main() {
    foo(*gen_foo());
    foo_indirect(*gen_foo());

    {
        let x: fmt::Display = *gen_foo();
        foo(x);
    }

    {
        let x: fmt::Display = *gen_foo();
        let y: fmt::Display = *gen_foo();
        foo(x);
        foo(y);
    }

    {
        let mut cnt: usize = 3;
        let x = loop {
            let x: fmt::Display = *gen_foo();
            if cnt == 0 {
                break x;
            } else {
                cnt -= 1;
            }
        };
        foo(x);
    }

    {
        let x: fmt::Display = *gen_foo();
        let x = if true {
            x
        } else {
            *gen_foo()
        };
        foo(x);
    }
}
