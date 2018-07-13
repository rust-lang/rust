// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Qux {}
struct A;
struct B;
impl Qux for A {}
impl Qux for B {}

struct Foo<T, U: Qux = A, V: Qux = B>(T, U, V);

struct foo;
struct bar;

fn want<T>(t: T) {}

fn have_usize(f: usize) {
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
}

fn have_foo(f: foo) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo(f: Foo<foo>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo_b(f: Foo<foo, B>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn have_foo_foo_b_a(f: Foo<foo, B, A>) {
    want::<usize>(f); //~ ERROR mismatched types
    want::<foo>(f); //~ ERROR mismatched types
    want::<bar>(f); //~ ERROR mismatched types
    want::<Foo<usize>>(f); //~ ERROR mismatched types
    want::<Foo<usize, B>>(f); //~ ERROR mismatched types
    want::<Foo<foo>>(f); //~ ERROR mismatched types
    want::<Foo<foo, B>>(f); //~ ERROR mismatched types
    want::<Foo<bar>>(f); //~ ERROR mismatched types
    want::<Foo<bar, B>>(f); //~ ERROR mismatched types
    want::<&Foo<foo>>(f); //~ ERROR mismatched types
    want::<&Foo<foo, B>>(f); //~ ERROR mismatched types
}

fn main() {}
