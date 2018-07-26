// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Just check if we don't get an ICE for the _S type.

#![feature(const_size_of)]

use std::cell::Cell;
use std::mem;

pub struct S {
    s: Cell<usize>
}

pub type _S = [usize; 0 - (mem::size_of::<S>() != 4) as usize];
