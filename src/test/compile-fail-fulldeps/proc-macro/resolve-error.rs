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

#![feature(proc_macro)]

#[macro_use]
extern crate derive_foo;
#[macro_use]
extern crate derive_clona;
extern crate attr_proc_macro;

use attr_proc_macro::attr_proc_macro;

macro_rules! FooWithLongNam {
    () => {}
}

#[derive(FooWithLongNan)]
//~^ ERROR cannot find derive macro `FooWithLongNan` in this scope
//~^^ HELP did you mean `FooWithLongName`?
struct Foo;

#[attr_proc_macra]
//~^ ERROR cannot find attribute macro `attr_proc_macra` in this scope
//~^^ HELP did you mean `attr_proc_macro`?
struct Bar;

#[FooWithLongNan]
//~^ ERROR cannot find attribute macro `FooWithLongNan` in this scope
struct Asdf;

#[derive(Dlone)]
//~^ ERROR cannot find derive macro `Dlone` in this scope
//~^^ HELP did you mean `Clone`?
struct A;

#[derive(Dlona)]
//~^ ERROR cannot find derive macro `Dlona` in this scope
//~^^ HELP did you mean `Clona`?
struct B;

#[derive(attr_proc_macra)]
//~^ ERROR cannot find derive macro `attr_proc_macra` in this scope
struct C;

fn main() {
    FooWithLongNama!();
    //~^ ERROR cannot find macro `FooWithLongNama!` in this scope
    //~^^ HELP did you mean `FooWithLongNam!`?

    attr_proc_macra!();
    //~^ ERROR cannot find macro `attr_proc_macra!` in this scope

    Dlona!();
    //~^ ERROR cannot find macro `Dlona!` in this scope
}
