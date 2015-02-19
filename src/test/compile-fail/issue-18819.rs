// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::marker::MarkerTrait;

trait Foo : MarkerTrait {
    type Item;
}

struct X;

impl Foo for X {
    type Item = bool;
}

fn print_x(_: &Foo, extra: &str) {
    println!("{}", extra);
}

fn main() {
    print_x(X);  //~error this function takes 2 parameters but 1 parameter was supplied
}
