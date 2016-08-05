// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<'a>(&'a str);
struct Buzz<'a, 'b>(&'a str, &'b str);

enum Bar {
    A,
    B,
    C,
}

struct Baz<'a, 'b, 'c> {
    foo: Foo,
    //~^ ERROR E0107
    //~| expected 1 lifetime parameter
    buzz: Buzz<'a>,
    //~^ ERROR E0107
    //~| expected 2 lifetime parameters
    bar: Bar<'a>,
    //~^ ERROR E0107
    //~| unexpected lifetime parameter
    foo2: Foo<'a, 'b, 'c>,
    //~^ ERROR E0107
    //~| 2 unexpected lifetime parameters
}

fn main() {
}
