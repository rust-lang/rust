// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

#![feature(macro_rules)]


pub trait bomb { fn boom(&self, Ident); }
pub struct S;
impl bomb for S { fn boom(&self, _: Ident) { } }

pub struct Ident { name: uint }

// macro_rules! int3( () => ( unsafe { asm!( "int3" ); } ) )
macro_rules! int3( () => ( { } ) )

fn Ident_new() -> Ident {
    int3!();
    Ident {name: 0x6789ABCD }
}

pub fn light_fuse(fld: Box<bomb>) {
    int3!();
    let f = || {
        int3!();
        fld.boom(Ident_new()); // *** 1
    };
    f();
}

pub fn main() {
    let b = box S as Box<bomb>;
    light_fuse(b);
}
