// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-more_struct_aliases

struct S;

trait Tr {
    type A;
}

fn f<T: Tr<A = S>>() {
    let _ = T::A {};
    //~^ ERROR `Self` and associated types in struct expressions and patterns are unstable
}

impl S {
    fn f() {
        let _ = Self {};
        //~^ ERROR `Self` and associated types in struct expressions and patterns are unstable
    }
}

fn main() {}
