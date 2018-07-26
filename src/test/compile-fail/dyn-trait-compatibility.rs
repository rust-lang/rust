// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type A0 = dyn;
//~^ ERROR cannot find type `dyn` in this scope
type A1 = dyn::dyn;
//~^ ERROR Use of undeclared type or module `dyn`
type A2 = dyn<dyn, dyn>;
//~^ ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
type A3 = dyn<<dyn as dyn>::dyn>;
//~^ ERROR cannot find type `dyn` in this scope
//~| ERROR cannot find type `dyn` in this scope
//~| ERROR Use of undeclared type or module `dyn`

fn main() {}
