// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 22443: Reject code using non-regular types that would
// otherwise cause dropck to loop infinitely.
//
// This version is just checking that we still sanely handle a trivial
// wrapper around the non-regular type. (It also demonstrates how the
// error messages will report different types depending on which type
// dropck is analyzing.)

use std::marker::PhantomData;

struct Digit<T> {
    elem: T
}

struct Node<T:'static> { m: PhantomData<&'static T> }

enum FingerTree<T:'static> {
    Single(T),
    // According to the bug report, Digit before Box would infinite loop.
    Deep(
        Digit<T>,
        Box<FingerTree<Node<T>>>,
        )
}

enum Wrapper<T:'static> {
    Simple,
    Other(FingerTree<T>),
}

fn main() {
    let w = //~ ERROR overflow while adding drop-check rules for std::option
        Some(Wrapper::Simple::<u32>);
    //~^ ERROR overflow while adding drop-check rules for std::option::Option
    //~| ERROR overflow while adding drop-check rules for Wrapper
}
