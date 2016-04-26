// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this will get a no-op Clone impl
#[derive(Copy, Clone)]
struct A {
    a: i32,
    b: i64
}

// this will get a deep Clone impl
#[derive(Copy, Clone)]
struct B<T> {
    a: i32,
    b: T
}

struct C; // not Copy or Clone
#[derive(Clone)] struct D; // Clone but not Copy

fn is_copy<T: Copy>(_: T) {}
fn is_clone<T: Clone>(_: T) {}

fn main() {
    // A can be copied and cloned
    is_copy(A { a: 1, b: 2 });
    is_clone(A { a: 1, b: 2 });

    // B<i32> can be copied and cloned
    is_copy(B { a: 1, b: 2 });
    is_clone(B { a: 1, b: 2 });

    // B<C> cannot be copied or cloned
    is_copy(B { a: 1, b: C }); //~ERROR Copy
    is_clone(B { a: 1, b: C }); //~ERROR Clone

    // B<D> can be cloned but not copied
    is_copy(B { a: 1, b: D }); //~ERROR Copy
    is_clone(B { a: 1, b: D });
}

