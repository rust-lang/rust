// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type A0 = auto;
//~^ ERROR cannot find type `auto` in this scope
type A1 = auto::auto;
//~^ ERROR Use of undeclared type or module `auto`
type A2 = auto<auto, auto>;
//~^ ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope
type A3 = auto<<auto as auto>::auto>;
//~^ ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope
//~| ERROR Use of undeclared type or module `auto`
type A4 = auto(auto, auto) -> auto;
//~^ ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope
//~| ERROR cannot find type `auto` in this scope

fn main() {}
