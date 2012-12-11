// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum box_impl<T> = {
    mut f: T
};

fn set_box_impl<T>(b: box_impl<@const T>, v: @const T) {
    b.f = v;
}

fn main() {
    let b = box_impl::<@int>({mut f: @3});
    set_box_impl(b, @mut 5);
    //~^ ERROR values differ in mutability

    // No error when type of parameter actually IS @const int
    let x: @const int = @3; // only way I could find to upcast
    let b = box_impl::<@const int>({mut f: x});
    set_box_impl(b, @mut 5);
}