// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Various tests related to testing how region inference works
// with respect to the object receivers.

trait Foo {
    fn borrowed<'a>(&'a self) -> &'a ();
}

// Here the receiver and return value all have the same lifetime,
// so no error results.
fn borrowed_receiver_same_lifetime<'a>(x: &'a Foo) -> &'a () {
    x.borrowed()
}

// Borrowed receiver but two distinct lifetimes, we get an error.
fn borrowed_receiver_different_lifetimes<'a,'b>(x: &'a Foo) -> &'b () {
    x.borrowed() //~ ERROR cannot infer
}

// Borrowed receiver with two distinct lifetimes, but we know that
// 'b:'a, hence &'a () is permitted.
fn borrowed_receiver_related_lifetimes<'a,'b>(x: &'a Foo+'b) -> &'a () {
    x.borrowed()
}

// Here we have two distinct lifetimes, but we try to return a pointer
// with the longer lifetime when (from the signature) we only know
// that it lives as long as the shorter lifetime. Therefore, error.
fn borrowed_receiver_related_lifetimes2<'a,'b>(x: &'a Foo+'b) -> &'b () {
    x.borrowed() //~ ERROR cannot infer
}

// Here, the object is bounded by an anonymous lifetime and returned
// as `&'static`, so you get an error.
fn owned_receiver(x: Box<Foo>) -> &'static () {
    x.borrowed() //~ ERROR cannot infer
}

fn main() {}

