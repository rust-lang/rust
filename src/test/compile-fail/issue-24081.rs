// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add; //~ ERROR import `Add` conflicts with type in this module
use std::ops::Sub; //~ ERROR import `Sub` conflicts with type in this module
use std::ops::Mul; //~ ERROR import `Mul` conflicts with type in this module
use std::ops::Div; //~ ERROR import `Div` conflicts with existing submodule
use std::ops::Rem; //~ ERROR import `Rem` conflicts with trait in this module

type Add = bool;
struct Sub { x: f32 }
enum Mul { A, B }
mod Div { }
trait Rem { }

fn main() {}
