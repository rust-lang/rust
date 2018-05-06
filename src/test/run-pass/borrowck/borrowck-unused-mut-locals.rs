// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![deny(unused_mut)]

#[derive(Debug)]
struct A {}

fn init_a() -> A {
    A {}
}

#[derive(Debug)]
struct B<'a> {
    ed: &'a mut A,
}

fn init_b<'a>(ed: &'a mut A) -> B<'a> {
    B { ed }
}

#[derive(Debug)]
struct C<'a> {
    pd: &'a mut B<'a>,
}

fn init_c<'a>(pd: &'a mut B<'a>) -> C<'a> {
    C { pd }
}

#[derive(Debug)]
struct D<'a> {
    sd: &'a mut C<'a>,
}

fn init_d<'a>(sd: &'a mut C<'a>) -> D<'a> {
    D { sd }
}

fn main() {
    let mut a = init_a();
    let mut b = init_b(&mut a);
    let mut c = init_c(&mut b);

    let d = init_d(&mut c);

    println!("{:?}", d)
}
