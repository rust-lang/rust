// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:internal_unstable.rs

#![feature(allow_internal_unstable)]

#[macro_use]
extern crate internal_unstable;

macro_rules! foo {
    ($e: expr, $f: expr) => {{
        $e;
        $f;
        internal_unstable::unstable(); //~ ERROR use of unstable
    }}
}

#[allow_internal_unstable]
macro_rules! bar {
    ($e: expr) => {{
        foo!($e,
             internal_unstable::unstable());
        internal_unstable::unstable();
    }}
}

fn main() {
    // ok, the instability is contained.
    call_unstable_allow!();
    construct_unstable_allow!(0);
    |x: internal_unstable::Foo| { call_method_allow!(x) };
    |x: internal_unstable::Bar| { access_field_allow!(x) };

    // bad.
    pass_through_allow!(internal_unstable::unstable()); //~ ERROR use of unstable

    pass_through_noallow!(internal_unstable::unstable()); //~ ERROR use of unstable



    println!("{:?}", internal_unstable::unstable()); //~ ERROR use of unstable

    bar!(internal_unstable::unstable()); //~ ERROR use of unstable
}
