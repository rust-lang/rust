// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for an ICE that occured when a default method implementation
// was applied to a type that did not meet the prerequisites. The
// problem occurred specifically because normalizing
// `Self::Item::Target` was impossible in this case.

use std::boxed::Box;
use std::marker::Sized;
use std::clone::Clone;
use std::ops::Deref;
use std::option::Option;
use std::option::Option::{Some,None};

trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    fn clone_first(mut self) -> Option<<Self::Item as Deref>::Target> where
        Self: Sized,
        Self::Item: Deref,
        <Self::Item as Deref>::Target: Clone,
    {
        self.next().cloned()
    }
}

struct Counter {
    value: i32
}

struct Token {
    value: i32
}

impl Iterator for Counter {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        let x = self.value;
        self.value += 1;
        Some(Token { value: x })
    }
}

fn main() {
    let mut x: Box<Iterator<Item=Token>> = Box::new(Counter { value: 22 });
    assert_eq!(x.next().unwrap().value, 22);
}
