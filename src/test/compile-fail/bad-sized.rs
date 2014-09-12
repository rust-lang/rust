// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

use std::cell::RefCell;

trait Trait {}

pub fn main() {
    let x: Vec<Trait + Sized> = Vec::new();
    //~^ ERROR the trait `core::kinds::Sized` is not implemented
    //~^^ ERROR the trait `core::kinds::Sized` is not implemented
    let x: Vec<Box<RefCell<Trait + Sized>>> = Vec::new();
    //~^ ERROR the trait `core::kinds::Sized` is not implemented
}
