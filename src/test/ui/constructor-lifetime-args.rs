// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// All lifetime parameters in struct constructors are currently considered early bound,
// i.e. `S::<ARGS>` is interpreted kinda like an associated item `S::<ARGS>::ctor`.
// This behavior is a bit weird, because if equivalent constructor were written manually
// it would get late bound lifetime parameters.
// Variant constructors behave in the same way, lifetime parameters are considered
// belonging to the enum and being early bound.
// https://github.com/rust-lang/rust/issues/30904

struct S<'a, 'b>(&'a u8, &'b u8);
enum E<'a, 'b> {
    V(&'a u8),
    U(&'b u8),
}

fn main() {
    S(&0, &0); // OK
    S::<'static>(&0, &0);
    //~^ ERROR expected 2 lifetime parameters, found 1 lifetime parameter
    S::<'static, 'static, 'static>(&0, &0);
    //~^ ERROR expected at most 2 lifetime parameters, found 3 lifetime parameters
    E::V(&0); // OK
    E::V::<'static>(&0);
    //~^ ERROR expected 2 lifetime parameters, found 1 lifetime parameter
    E::V::<'static, 'static, 'static>(&0);
    //~^ ERROR expected at most 2 lifetime parameters, found 3 lifetime parameters
}
