// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// aux-build:trait_superkinds_in_metadata.rs

// Test for traits inheriting from the builtin kinds cross-crate.
// Mostly tests correctness of metadata.

extern crate trait_superkinds_in_metadata;
use trait_superkinds_in_metadata::{RequiresRequiresShareAndSend, RequiresShare};

struct X<T>(T);

impl <T:Sync> RequiresShare for X<T> { }

impl <T:Sync+'static> RequiresRequiresShareAndSend for X<T> { }
//~^ ERROR the trait `core::marker::Send` is not implemented

fn main() { }
