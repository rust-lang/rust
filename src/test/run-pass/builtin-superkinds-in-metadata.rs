// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

// aux-build:trait_superkinds_in_metadata.rs

// Tests (correct) usage of trait super-builtin-kinds cross-crate.

extern mod trait_superkinds_in_metadata;
use trait_superkinds_in_metadata::{RequiresRequiresFreezeAndSend, RequiresFreeze};
use trait_superkinds_in_metadata::{RequiresPod};

struct X<T>(T);

impl <T:Freeze> RequiresFreeze for X<T> { }

impl <T:Freeze+Send> RequiresRequiresFreezeAndSend for X<T> { }

impl <T:Pod> RequiresPod for X<T> { }

pub fn main() { }
