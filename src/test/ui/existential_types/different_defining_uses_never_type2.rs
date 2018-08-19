// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(existential_type)]

fn main() {}

// two definitions with different types
existential type Foo: std::fmt::Debug;

fn foo() -> Foo {
    ""
}

fn bar(arg: bool) -> Foo {
    if arg {
        panic!()
    } else {
        "bar"
    }
}

fn boo(arg: bool) -> Foo {
    if arg {
        loop {}
    } else {
        "boo"
    }
}

fn bar2(arg: bool) -> Foo {
    if arg {
        "bar2"
    } else {
        panic!()
    }
}

fn boo2(arg: bool) -> Foo {
    if arg {
        "boo2"
    } else {
        loop {}
    }
}
