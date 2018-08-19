// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// Check that global bounds result in the expected choice of associated type

#![feature(trivial_bounds)]
#![allow(unused)]

struct B;

trait A {
    type X;
    fn get_x() -> Self::X;
}

impl A for B {
    type X = u8;
    fn get_x() -> u8 { 0 }
}

fn underspecified_bound() -> u8
where
    B: A
{
    B::get_x()
}

fn inconsistent_bound() -> i32
where
    B: A<X = i32>
{
    B::get_x()
}

fn redundant_bound() -> u8
where
    B: A<X = u8>
{
    B::get_x()
}

fn inconsistent_dup_bound() -> i32
where
    B: A<X = i32> + A
{
    B::get_x()
}

fn redundant_dup_bound() -> u8
where
    B: A<X = u8> + A
{
    B::get_x()
}

fn main () {}
