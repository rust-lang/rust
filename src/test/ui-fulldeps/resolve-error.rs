// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-foo.rs
// aux-build:derive-clona.rs
// aux-build:attr_proc_macro.rs
// aux-build:bang_proc_macro.rs

#![feature(proc_macro)]

#[macro_use]
extern crate derive_foo;
#[macro_use]
extern crate derive_clona;
extern crate attr_proc_macro;
extern crate bang_proc_macro;

use attr_proc_macro::attr_proc_macro;
use bang_proc_macro::bang_proc_macro;

macro_rules! FooWithLongNam {
    () => {}
}

macro_rules! attr_proc_mac {
    () => {}
}

#[derive(FooWithLongNan)]
//~^ ERROR cannot find
struct Foo;

#[attr_proc_macra]
//~^ ERROR cannot find
struct Bar;

#[FooWithLongNan]
//~^ ERROR cannot find
struct Asdf;

#[derive(Dlone)]
//~^ ERROR cannot find
struct A;

#[derive(Dlona)]
//~^ ERROR cannot find
struct B;

#[derive(attr_proc_macra)]
//~^ ERROR cannot find
struct C;

fn main() {
    FooWithLongNama!();
    //~^ ERROR cannot find

    attr_proc_macra!();
    //~^ ERROR cannot find

    Dlona!();
    //~^ ERROR cannot find

    bang_proc_macrp!();
    //~^ ERROR cannot find
}
