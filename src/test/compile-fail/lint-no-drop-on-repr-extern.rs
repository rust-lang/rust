// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check we reject structs that mix a `Drop` impl with `#[repr(C)]`.
//
// As a special case, also check that we do not warn on such structs
// if they also are declared with `#[unsafe_no_drop_flag]`

#![feature(unsafe_no_drop_flag)]
#![deny(drop_with_repr_extern)]
//~^ NOTE lint level defined here
//~| NOTE lint level defined here

#[repr(C)] struct As { x: Box<i8> }
#[repr(C)] enum Ae { Ae(Box<i8>), _None }

struct Bs { x: Box<i8> }
enum Be { Be(Box<i8>), _None }

#[repr(C)] struct Cs { x: Box<i8> }
//~^ NOTE the `#[repr(C)]` attribute is attached here

impl Drop for Cs { fn drop(&mut self) { } }
//~^ ERROR implementing Drop adds hidden state to types, possibly conflicting with `#[repr(C)]`

#[repr(C)] enum Ce { Ce(Box<i8>), _None }
//~^ NOTE the `#[repr(C)]` attribute is attached here

impl Drop for Ce { fn drop(&mut self) { } }
//~^ ERROR implementing Drop adds hidden state to types, possibly conflicting with `#[repr(C)]`

#[unsafe_no_drop_flag]
#[repr(C)] struct Ds { x: Box<i8> }

impl Drop for Ds { fn drop(&mut self) { } }

#[unsafe_no_drop_flag]
#[repr(C)] enum De { De(Box<i8>), _None }

impl Drop for De { fn drop(&mut self) { } }

fn main() {
    let a = As { x: Box::new(3) };
    let b = Bs { x: Box::new(3) };
    let c = Cs { x: Box::new(3) };
    let d = Ds { x: Box::new(3) };

    println!("{:?}", (*a.x, *b.x, *c.x, *d.x));

    let _a = Ae::Ae(Box::new(3));
    let _b = Be::Be(Box::new(3));
    let _c = Ce::Ce(Box::new(3));
    let _d = De::De(Box::new(3));
}
