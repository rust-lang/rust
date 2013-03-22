// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving_clone] //~ ERROR `#[deriving_clone]` is obsolete; use `#[deriving(Clone)]` instead
#[deriving_eq] //~ ERROR `#[deriving_eq]` is obsolete; use `#[deriving(Eq)]` instead
#[deriving_iter_bytes]
//~^ ERROR `#[deriving_iter_bytes]` is obsolete; use `#[deriving(IterBytes)]` instead
struct Foo;

pub fn main() { }
