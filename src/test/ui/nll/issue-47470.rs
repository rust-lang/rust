// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #47470: cached results of projections were
// causing region relations not to be enforced at all the places where
// they have to be enforced.

#![feature(nll)]

struct Foo<'a>(&'a ());
trait Bar {
    type Assoc;
    fn get(self) -> Self::Assoc;
}

impl<'a> Bar for Foo<'a> {
    type Assoc = &'a u32;
    fn get(self) -> Self::Assoc {
        let local = 42;
        &local //~ ERROR `local` does not live long enough
    }
}

fn main() {
    let f = Foo(&()).get();
    println!("{}", f);
}
