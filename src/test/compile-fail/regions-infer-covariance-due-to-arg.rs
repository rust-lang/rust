// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Covariant with respect to a region:
//
// You can upcast to a *larger region* but not a smaller one.

struct covariant<'self> {
    f: &'static fn(x: &'self int) -> int
}

fn to_same_lifetime<'r>(bi: covariant<'r>) {
    let bj: covariant<'r> = bi;
}

fn to_shorter_lifetime<'r>(bi: covariant<'r>) {
    let bj: covariant<'blk> = bi; //~ ERROR mismatched types
    //~^ ERROR cannot infer an appropriate lifetime
}

fn to_longer_lifetime<'r>(bi: covariant<'r>) -> covariant<'static> {
    bi
}

fn main() {
}
