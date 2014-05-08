// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that genuine type errors with partial
// type hints are understandable.

struct Foo<T>;
struct Bar<U>;

pub fn main() {
}

fn test1() {
    let x: Foo<_> = Bar::<uint>;
    //~^ ERROR mismatched types: expected `Foo<<generic #0>>` but found `Bar<uint>`
    let y: Foo<uint> = x;
}

fn test2() {
    let x: Foo<_> = Bar::<uint>;
    //~^ ERROR mismatched types: expected `Foo<<generic #0>>` but found `Bar<uint>`
}
