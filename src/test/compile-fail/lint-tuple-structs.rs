// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(using_tuple_structs)]
#![allow(dead_code)]

// regular struct is okay
pub struct S { x: usize, y: usize }

// enum-like struct is okay, too
pub struct ES;

// tuple-like struct is not
pub struct TS(usize);   //~ ERROR standard library should not use tuple-like

// but non-public one is
struct TSPrivate(usize);

fn main() {}
