// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `impl MyTrait for &i32` works and is equivalent to any lifetime.

// run-pass

#![allow(warnings)]

#![feature(in_band_lifetimes)]

trait MyTrait { }

impl MyTrait for &i32 {
}

fn impls_my_trait<T: MyTrait>() { }

fn impls_my_trait_val<T: MyTrait>(_: T) {
    impls_my_trait::<T>();
}

fn random_where_clause()
where for<'a> &'a i32: MyTrait { }

fn main() {
    let x = 22;
    let f = &x;

    impls_my_trait_val(f);

    impls_my_trait::<&'static i32>();

    random_where_clause();
}
