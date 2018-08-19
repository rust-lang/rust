// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// normalize-stderr-test: "The system cannot find the file specified\." -> "No such file or directory"
// ignore-tidy-linelength

#![feature(external_doc)]

#[doc(include = "not-a-file.md")] //~ ERROR: couldn't read
pub struct SomeStruct;

fn main() {}
