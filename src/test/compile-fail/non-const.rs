// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that various non const things are rejected.

fn foo<T: Const>(_x: T) { }

struct r {
  x:int,
}

impl r : Drop {
    fn finalize(&self) {}
}

fn r(x:int) -> r {
    r {
        x: x
    }
}

struct r2 {
  x:@mut int,
}

impl r2 : Drop {
    fn finalize(&self) {}
}

fn r2(x:@mut int) -> r2 {
    r2 {
        x: x
    }
}

fn main() {
    foo({f: 3});
    foo({mut f: 3}); //~ ERROR missing `const`
    foo(~[1]);
    foo(~[1]); //~ ERROR missing `const`
    foo(~1);
    foo(~mut 1); //~ ERROR missing `const`
    foo(@1);
    foo(@mut 1); //~ ERROR missing `const`
    foo(r(1)); // this is okay now.
    foo(r2(@mut 1)); //~ ERROR missing `const`
    foo({f: {mut f: 1}}); //~ ERROR missing `const`
}
