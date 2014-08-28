// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;

trait Trait {}

pub fn main() {
    let x: Vec<Trait + Sized> = Vec::new();
    //~^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
    //~^^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
    let x: Vec<Box<Trait + Sized>> = Vec::new();
    //~^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
    //~^^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
    let x: Vec<Box<RefCell<Trait + Sized>>> = Vec::new();
    //~^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
    //~^^ ERROR explicitly adding `Sized` bound to an unsized type `Trait+Sized`
}
