// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add; //~ NOTE previous import of the trait `Add` here
use std::ops::Sub; //~ NOTE previous import of the trait `Sub` here
use std::ops::Mul; //~ NOTE previous import of the trait `Mul` here
use std::ops::Div; //~ NOTE previous import of the trait `Div` here
use std::ops::Rem; //~ NOTE previous import of the trait `Rem` here

type Add = bool; //~ ERROR the name `Add` is defined multiple times
//~| `Add` redefined here
//~| NOTE `Add` must be defined only once in the type namespace of this module
struct Sub { x: f32 } //~ ERROR the name `Sub` is defined multiple times
//~| `Sub` redefined here
//~| NOTE `Sub` must be defined only once in the type namespace of this module
enum Mul { A, B } //~ ERROR the name `Mul` is defined multiple times
//~| `Mul` redefined here
//~| NOTE `Mul` must be defined only once in the type namespace of this module
mod Div { } //~ ERROR the name `Div` is defined multiple times
//~| `Div` redefined here
//~| NOTE `Div` must be defined only once in the type namespace of this module
trait Rem {  } //~ ERROR the name `Rem` is defined multiple times
//~| `Rem` redefined here
//~| NOTE `Rem` must be defined only once in the type namespace of this module

fn main() {}
