// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verify that Unsafe is *always* share regardles `T` is share.

// ignore-tidy-linelength

use std::ty::Unsafe;
use std::kinds::marker;

struct MyShare<T> {
    u: Unsafe<T>
}

struct NoShare {
    m: marker::NoShare
}

fn test<T: Share>(s: T){

}

fn main() {
    let us = Unsafe::new(MyShare{u: Unsafe::new(0)});
    test(us);

    let uns = Unsafe::new(NoShare{m: marker::NoShare});
    test(uns);

    let ms = MyShare{u: uns};
    test(ms);

    let ns = NoShare{m: marker::NoShare};
    test(ns);
    //~^ ERROR instantiating a type parameter with an incompatible type `NoShare`, which does not fulfill `Share`
}
