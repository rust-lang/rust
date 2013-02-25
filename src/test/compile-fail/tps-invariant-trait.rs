// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait box_trait<T> {
    fn get() -> T;
    fn set(t: T);
}

struct box<T> {
    f: T
}

enum box_impl<T> = box<T>;

impl<T:Copy> box_trait<T> for box_impl<T> {
    fn get() -> T { return self.f; }
    fn set(t: T) { self.f = t; }
}

fn set_box_trait<T>(b: box_trait<@const T>, v: @const T) {
    b.set(v);
}

fn set_box_impl<T>(b: box_impl<@const T>, v: @const T) {
    b.set(v);
}

fn main() {
    let b = box_impl::<@int>(box::<@int> {f: @3});
    set_box_trait(b as box_trait::<@int>, @mut 5);
    //~^ ERROR values differ in mutability
    set_box_impl(b, @mut 5);
    //~^ ERROR values differ in mutability
}
