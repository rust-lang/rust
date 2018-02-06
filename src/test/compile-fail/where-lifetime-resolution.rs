// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait1<'a> {}
trait Trait2<'a, 'b> {}

fn f() where
    for<'a> Trait1<'a>: Trait1<'a>, // OK
    (for<'a> Trait1<'a>): Trait1<'a>,
    //~^ ERROR use of undeclared lifetime name `'a`
    for<'a> for<'b> Trait2<'a, 'b>: Trait2<'a, 'b>,
    //~^ ERROR use of undeclared lifetime name `'b`
    //~| ERROR nested quantification of lifetimes
{}

fn main() {}
