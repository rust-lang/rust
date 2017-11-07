// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// revisions: mir
//[mir]compile-flags: -Z emit-end-regions -Z borrowck-mir

struct Wrap<'p> { p: &'p mut i32 }

impl<'p> Drop for Wrap<'p> {
    fn drop(&mut self) {
        *self.p += 1;
    }
}

fn foo() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    x += 1; //~ ERROR because of dtor
}

fn bar() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    std::mem::drop(wrap);
    x += 1; // OK, drop is inert
}

struct Foo<'p> { a: String, b: Wrap<'p> }

fn move_string(_a: String) { }

fn move_wrap<'p>(_b: Wrap<'p>) { }

fn baz_a() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    let s = String::from("str");
    let foo = Foo { a: s, b: wrap };
    move_string(foo.a);
}

fn baz_a_b() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    let s = String::from("str");
    let foo = Foo { a: s, b: wrap };
    move_string(foo.a);
    move_wrap(foo.b);
}

fn baz_b() {
    let mut x = 0;
    let wrap = Wrap { p: &mut x };
    let s = String::from("str");
    let foo = Foo { a: s, b: wrap };
    move_wrap(foo.b);
}

fn main() { }