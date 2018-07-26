// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Zmir-opt-level=1

use std::mem;

#[derive(Copy, Clone)]
enum Never {}

union Foo {
    a: u64,
    b: Never
}

fn foo(xs: [(Never, u32); 1]) -> u32 { xs[0].1 }

fn bar([(_, x)]: [(Never, u32); 1]) -> u32 { x }

fn main() {
    println!("{}", mem::size_of::<Foo>());

    let f = [Foo { a: 42 }, Foo { a: 10 }];
    println!("{:?}", unsafe { f[0].a });
}
