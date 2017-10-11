// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    f1(|_: (), _: ()| {});
    f2(|_: (), _: ()| {});
    f3(|_: (), _: ()| {});
    f4(|_: (), _: ()| {});
    f5(|_: (), _: ()| {});
    g1(|_: (), _: ()| {});
    g2(|_: (), _: ()| {});
    g3(|_: (), _: ()| {});
    g4(|_: (), _: ()| {});
    h1(|_: (), _: (), _: (), _: ()| {});
    h2(|_: (), _: (), _: (), _: ()| {});
}

// Basic
fn f1<F>(_: F) where F: Fn(&(), &()) {}
fn f2<F>(_: F) where F: for<'a> Fn(&'a (), &()) {}
fn f3<'a, F>(_: F) where F: Fn(&'a (), &()) {}
fn f4<F>(_: F) where F: for<'r> Fn(&(), &'r ()) {}
fn f5<F>(_: F) where F: for<'r> Fn(&'r (), &'r ()) {}

// Nested
fn g1<F>(_: F) where F: Fn(&(), Box<Fn(&())>) {}
fn g2<F>(_: F) where F: Fn(&(), fn(&())) {}
fn g3<F>(_: F) where F: for<'s> Fn(&'s (), Box<Fn(&())>) {}
fn g4<F>(_: F) where F: Fn(&(), for<'r> fn(&'r ())) {}

// Mixed
fn h1<F>(_: F) where F: Fn(&(), Box<Fn(&())>, &(), fn(&(), &())) {}
fn h2<F>(_: F) where F: for<'t0> Fn(&(), Box<Fn(&())>, &'t0 (), fn(&(), &())) {}
