//@ run-pass
#![allow(unused_imports)]
// Test for an ICE that occurred when a default method implementation
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

    fn clone_first(mut self) -> Option<<Self::Item as Deref>::Target> where //~ WARN method `clone_first` is never used
        Self: Sized,
        Self::Item: Deref,
        <Self::Item as Deref>::Target: Clone,
    {
        self.next().map(|x| x.clone())
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
    let mut x: Box<dyn Iterator<Item=Token>> = Box::new(Counter { value: 22 });
    assert_eq!(x.next().unwrap().value, 22);
}
