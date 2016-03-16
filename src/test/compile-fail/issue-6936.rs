// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct T;

mod t1 {
    type Foo = ::T;
    mod Foo {} //~ ERROR: `Foo` has already been defined
}

mod t2 {
    type Foo = ::T;
    struct Foo; //~ ERROR: `Foo` has already been defined
}

mod t3 {
    type Foo = ::T;
    enum Foo {} //~ ERROR: `Foo` has already been defined
}

mod t4 {
    type Foo = ::T;
    fn Foo() {} // ok
}

mod t5 {
    type Bar<T> = T;
    mod Bar {} //~ ERROR: `Bar` has already been defined
}

mod t6 {
    type Foo = ::T;
    impl Foo {} // ok
}


fn main() {}
