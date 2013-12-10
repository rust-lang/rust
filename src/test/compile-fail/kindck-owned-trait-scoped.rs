// xfail-test
// xfail'd because to_foo() doesn't work.

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A dummy trait/impl that work close over any type.  The trait will
// be parameterized by a region due to the &'a int constraint.

trait foo {
    fn foo(&self, i: &'a int) -> int;
}

impl<T:Clone> foo for T {
    fn foo(&self, i: &'a int) -> int {*i}
}

fn to_foo<T:Clone>(t: T) {
    // This version is ok because, although T may contain borrowed
    // pointers, it never escapes the fn body.  We know this because
    // the type of foo includes a region which will be resolved to
    // the fn body itself.
    let v = &3;
    struct F<T> { f: T }
    let x = @F {f:t} as @foo;
    assert_eq!(x.foo(v), 3);
}

fn to_foo_2<T:Clone>(t: T) -> @foo {
    // Not OK---T may contain borrowed ptrs and it is going to escape
    // as part of the returned foo value
    struct F<T> { f: T }
    @F {f:t} as @foo //~ ERROR value may contain borrowed pointers; add `'static` bound
}

fn to_foo_3<T:Clone + 'static>(t: T) -> @foo {
    // OK---T may escape as part of the returned foo value, but it is
    // owned and hence does not contain borrowed ptrs
    struct F<T> { f: T }
    @F {f:t} as @foo
}

fn main() {
}
