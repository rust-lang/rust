// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

struct Bar<'x, 'y, 'z> { bar: &'y int, baz: int }
fn bar1<'a>(x: &Bar) -> (&'a int, &'a int, &'a int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar1<'b, 'c, 'a>(x: &'a Bar<'b, 'a, 'c>) -> (&'a int, &'a int, &'a int)
    (x.bar, &x.baz, &x.baz)
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
    //~^^^ ERROR: cannot infer
}

fn bar2<'a, 'b, 'c>(x: &Bar<'a, 'b, 'c>) -> (&'a int, &'a int, &'a int) {
//~^ NOTE: consider using an explicit lifetime parameter as shown: fn bar2<'a, 'c>(x: &'a Bar<'a, 'a, 'c>) -> (&'a int, &'a int, &'a int)
    (x.bar, &x.baz, &x.baz)
    //~^ ERROR: cannot infer
    //~^^ ERROR: cannot infer
    //~^^^ ERROR: cannot infer
}

fn main() { }
