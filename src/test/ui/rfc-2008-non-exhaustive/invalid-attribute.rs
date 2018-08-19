// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(non_exhaustive)]

#[non_exhaustive(anything)]
//~^ ERROR attribute should be empty [E0702]
struct Foo;

#[non_exhaustive]
//~^ ERROR attribute can only be applied to a struct or enum [E0701]
trait Bar { }

#[non_exhaustive]
//~^ ERROR attribute can only be applied to a struct or enum [E0701]
union Baz {
    f1: u16,
    f2: u16
}

fn main() { }
