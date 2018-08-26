// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![crate_type = "lib"]
#![feature(const_panic)]

const Z: () = panic!("cheese");
//~^ ERROR any use of this value will cause an error

const Y: () = unreachable!();
//~^ ERROR any use of this value will cause an error

const X: () = unimplemented!();
//~^ ERROR any use of this value will cause an error
