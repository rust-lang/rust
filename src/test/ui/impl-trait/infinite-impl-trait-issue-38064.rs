// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that attempts to construct infinite types via impl trait fail
// in a graceful way.
//
// Regression test for #38064.

// error-pattern:overflow evaluating the requirement `impl Quux`

trait Quux {}

fn foo() -> impl Quux {
    struct Foo<T>(T);
    impl<T> Quux for Foo<T> {}
    Foo(bar())
}

fn bar() -> impl Quux {
    struct Bar<T>(T);
    impl<T> Quux for Bar<T> {}
    Bar(foo())
}

// effectively:
//     struct Foo(Bar);
//     struct Bar(Foo);
// should produce an error about infinite size

fn main() { foo(); }
