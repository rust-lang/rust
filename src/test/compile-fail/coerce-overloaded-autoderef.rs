// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn borrow_mut<T>(x: &mut T) -> &mut T { x }
fn borrow<T>(x: &T) -> &T { x }

fn borrow_mut2<T>(_: &mut T, _: &mut T) {}
fn borrow2<T>(_: &mut T, _: &T) {}

fn double_mut_borrow<T>(x: &mut Box<T>) {
    let y = borrow_mut(x);
    let z = borrow_mut(x);
    //~^ ERROR cannot borrow `*x` as mutable more than once at a time
}

fn double_imm_borrow(x: &mut Box<i32>) {
    let y = borrow(x);
    let z = borrow(x);
    **x += 1;
    //~^ ERROR cannot assign to `**x` because it is borrowed
}

fn double_mut_borrow2<T>(x: &mut Box<T>) {
    borrow_mut2(x, x);
    //~^ ERROR cannot borrow `*x` as mutable more than once at a time
}

fn double_borrow2<T>(x: &mut Box<T>) {
    borrow2(x, x);
    //~^ ERROR cannot borrow `*x` as immutable because it is also borrowed as mutable
}

pub fn main() {}
