// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod m {}
enum E {}
struct S;
trait Tr {}

use {}; // OK
use ::{}; // OK
use m::{}; // OK
use E::{}; // OK
use S::{}; //~ ERROR expected module or enum, found struct `S`
use Tr::{}; //~ ERROR expected module or enum, found trait `Tr`
use Nonexistent::{}; //~ ERROR cannot find module or enum `Nonexistent` in the crate root

fn main () {}
