// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-trait_alias

trait Alias1<T> = Default where T: Clone; // ok
    //~^ERROR trait aliases are not yet fully implemented
trait Alias2<T: Clone = ()> = Default;
    //~^ERROR type parameters on the left side of a trait alias cannot be bounded
    //~^^ERROR type parameters on the left side of a trait alias cannot have defaults
    //~^^^ERROR trait aliases are not yet fully implemented

impl Alias1 { //~ERROR expected type, found trait alias
}

impl Alias1 for () { //~ERROR expected trait, found trait alias
}

fn main() {}

