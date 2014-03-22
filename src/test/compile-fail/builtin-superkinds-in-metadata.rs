// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

// aux-build:trait_superkinds_in_metadata.rs

// Test for traits inheriting from the builtin kinds cross-crate.
// Mostly tests correctness of metadata.

extern crate trait_superkinds_in_metadata;
use trait_superkinds_in_metadata::{RequiresRequiresShareAndSend, RequiresShare};

struct X<T>(T);

impl <T:Share> RequiresShare for X<T> { }

impl <T:Share> RequiresRequiresShareAndSend for X<T> { } //~ ERROR cannot implement this trait

fn main() { }
