// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

#![recursion_limit = "32"]

#[derive(Clone)]
struct A (B);

impl A {
    pub fn matches<F: Fn()>(&self, f: &F) {
        //~^ ERROR reached the recursion limit during monomorphization
        let &A(ref term) = self;
        term.matches(f);
    }
}

#[derive(Clone)]
enum B {
    Variant1,
    Variant2(C),
}

impl B {
    pub fn matches<F: Fn()>(&self, f: &F) {
        match self {
            &B::Variant2(ref factor) => {
                factor.matches(&|| ())
            }
            _ => unreachable!("")
        }
    }
}

#[derive(Clone)]
struct C (D);

impl C {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let &C(ref base) = self;
        base.matches(&|| {
            C(base.clone()).matches(f)
        })
    }
}

#[derive(Clone)]
struct D (Box<A>);

impl D {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let &D(ref a) = self;
        a.matches(f)
    }
}

pub fn matches() {
    A(B::Variant1).matches(&(|| ()))
}

fn main() {}
