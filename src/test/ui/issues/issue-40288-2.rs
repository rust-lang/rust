// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn prove_static<T: 'static + ?Sized>(_: &'static T) {}

fn lifetime_transmute_slice<'a, T: ?Sized>(x: &'a T, y: &T) -> &'a T {
    let mut out = [x];
    {
        let slice: &mut [_] = &mut out;
        slice[0] = y;
    }
    out[0]
    //~^ ERROR 19:5: 19:11: explicit lifetime required in the type of `y` [E0621]
}

struct Struct<T, U: ?Sized> {
    head: T,
    _tail: U
}

fn lifetime_transmute_struct<'a, T: ?Sized>(x: &'a T, y: &T) -> &'a T {
    let mut out = Struct { head: x, _tail: [()] };
    {
        let dst: &mut Struct<_, [()]> = &mut out;
        dst.head = y;
    }
    out.head
    //~^ ERROR 34:5: 34:13: explicit lifetime required in the type of `y` [E0621]
}

fn main() {
    prove_static(lifetime_transmute_slice("", &String::from("foo")));
    prove_static(lifetime_transmute_struct("", &String::from("bar")));
}
