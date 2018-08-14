// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

// Test that we still see borrowck errors of various kinds when using
// indexing and autoderef in combination.

use std::ops::{Index, IndexMut};

struct Foo {
    x: isize,
    y: isize,
}

impl<'a> Index<&'a String> for Foo {
    type Output = isize;

    fn index(&self, z: &String) -> &isize {
        if *z == "x" {
            &self.x
        } else {
            &self.y
        }
    }
}

impl<'a> IndexMut<&'a String> for Foo {
    fn index_mut(&mut self, z: &String) -> &mut isize {
        if *z == "x" {
            &mut self.x
        } else {
            &mut self.y
        }
    }
}

fn test1(mut f: Box<Foo>, s: String) {
    let _p = &mut f[&s];
    let _q = &f[&s]; //~ ERROR cannot borrow
}

fn test2(mut f: Box<Foo>, s: String) {
    let _p = &mut f[&s];
    let _q = &mut f[&s]; //~ ERROR cannot borrow
}

struct Bar {
    foo: Foo
}

fn test3(mut f: Box<Bar>, s: String) {
    let _p = &mut f.foo[&s];
    let _q = &mut f.foo[&s]; //~ ERROR cannot borrow
}

fn test4(mut f: Box<Bar>, s: String) {
    let _p = &f.foo[&s];
    let _q = &f.foo[&s];
}

fn test5(mut f: Box<Bar>, s: String) {
    let _p = &f.foo[&s];
    let _q = &mut f.foo[&s]; //~ ERROR cannot borrow
}

fn test6(mut f: Box<Bar>, g: Foo, s: String) {
    let _p = &f.foo[&s];
    f.foo = g; //~ ERROR cannot assign
}

fn test7(mut f: Box<Bar>, g: Bar, s: String) {
    let _p = &f.foo[&s];
    *f = g; //~ ERROR cannot assign
}

fn test8(mut f: Box<Bar>, g: Foo, s: String) {
    let _p = &mut f.foo[&s];
    f.foo = g; //~ ERROR cannot assign
}

fn test9(mut f: Box<Bar>, g: Bar, s: String) {
    let _p = &mut f.foo[&s];
    *f = g; //~ ERROR cannot assign
}

fn main() {
}
