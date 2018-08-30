// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type Func = fn(usize, usize) -> usize;

fn foo(a: usize, b: usize) -> usize { a + b }
fn bar(a: usize, b: usize) -> usize { a * b }
fn test(x: usize) -> Func {
    if x % 2 == 0 { foo }
    else { bar }
}

const FOO: Func = foo;
const BAR: Func = bar;

fn main() {
    match test(std::env::consts::ARCH.len()) {
        FOO => println!("foo"),
        BAR => println!("bar"),
        _ => unreachable!(),
    }
}
