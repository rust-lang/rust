// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;
use std::ops::Rem;

type Add = bool; //~ ERROR the name `Add` is defined multiple times
//~| `Add` redefined here
struct Sub { x: f32 } //~ ERROR the name `Sub` is defined multiple times
//~| `Sub` redefined here
enum Mul { A, B } //~ ERROR the name `Mul` is defined multiple times
//~| `Mul` redefined here
mod Div { } //~ ERROR the name `Div` is defined multiple times
//~| `Div` redefined here
trait Rem {  } //~ ERROR the name `Rem` is defined multiple times
//~| `Rem` redefined here

fn main() {}
