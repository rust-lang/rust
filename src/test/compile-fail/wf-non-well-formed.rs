// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

// Check that we catch attempts to create non-well-formed types

trait Tr {}

fn g1<X: ?Sized>(x: X) {} //~ERROR the trait `core::marker::Sized` is not implemented
fn g2<X: ?Sized + Tr>(x: X) {} //~ERROR the trait `core::marker::Sized` is not implemented

fn bogus( //~ERROR the trait `core::marker::Sized` is not implemented
    _: [u8])
{
    loop {}
}

struct S<T: Tr>(T);

fn b1() -> &'static [(u8,fn(&'static [S<()>]))]
{} //~^ ERROR the trait `Tr` is not implemented
fn b2() -> fn([u8])
{} //~^ ERROR the trait `core::marker::Sized` is not implemented
fn b3() -> fn()->[u8]
{} //~^ ERROR the trait `core::marker::Sized` is not implemented
fn b4() -> &'static [[u8]]
{} //~^ ERROR the trait `core::marker::Sized` is not implemented
fn b5() -> &'static [[u8]; 2]
{} //~^ ERROR the trait `core::marker::Sized` is not implemented
fn b6() -> &'static ([u8],)
{} //~^ ERROR the trait `core::marker::Sized` is not implemented
fn b7() -> &'static (u32,[u8],u32)
{} //~^ ERROR the trait `core::marker::Sized` is not implemented

fn main() {}

trait TrBogus {
    const X1: &'static [[u16]];
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}

impl<T: Tr> Iterator for S<T> {
    type Item = ([u8], T);
    //~^ ERROR the trait `core::marker::Sized` is not implemented
}
