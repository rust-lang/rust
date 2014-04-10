// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A {}

struct Struct {
    r: A //~ ERROR reference to trait `A` where a type is expected; try `~A` or `&A`
}

fn new_struct(r: A) -> Struct {
    //~^ ERROR reference to trait `A` where a type is expected; try `~A` or `&A`
    Struct { r: r }
}

trait Curve {}
enum E {X(Curve)}
//~^ ERROR reference to trait `Curve` where a type is expected; try `~Curve` or `&Curve`
fn main() {}
