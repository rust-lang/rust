// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that parenthetical notation is feature-gated except with the
// `Fn` traits.

trait Foo<A,R> {
}

fn main() {
    let x: Box<Foo(int)>;
    //~^ ERROR parenthetical notation is only stable when used with the `Fn` family

    // No errors with these:
    let x: Box<Fn(int)>;
    let x: Box<FnMut(int)>;
    let x: Box<FnOnce(int)>;
}
