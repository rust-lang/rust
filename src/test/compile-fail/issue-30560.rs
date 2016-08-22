// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type Alias = ();
use Alias::*;
//~^ ERROR unresolved import `Alias::*` [E0432]
//~| Not a module `Alias`
use std::io::Result::*;
//~^ ERROR unresolved import `std::io::Result::*` [E0432]
//~| Not a module `Result`

trait T {}
use T::*; //~ ERROR items in traits are not importable

fn main() {}
