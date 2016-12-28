// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add; //~ NOTE previous import
use std::ops::Sub; //~ NOTE previous import
use std::ops::Mul; //~ NOTE previous import
use std::ops::Div; //~ NOTE previous import
use std::ops::Rem; //~ NOTE previous import

type Add = bool; //~ ERROR a trait named `Add` has already been imported in this module
//~| `Add` already imported
struct Sub { x: f32 } //~ ERROR a trait named `Sub` has already been imported in this module
//~| `Sub` already imported
enum Mul { A, B } //~ ERROR a trait named `Mul` has already been imported in this module
//~| `Mul` already imported
mod Div { } //~ ERROR a trait named `Div` has already been imported in this module
//~| `Div` already imported
trait Rem {  } //~ ERROR a trait named `Rem` has already been imported in this module
//~| `Rem` already imported

fn main() {}
