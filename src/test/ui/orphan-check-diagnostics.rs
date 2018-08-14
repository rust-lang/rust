// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:orphan_check_diagnostics.rs
// see #22388

extern crate orphan_check_diagnostics;

use orphan_check_diagnostics::RemoteTrait;

trait LocalTrait { fn dummy(&self) { } }

impl<T> RemoteTrait for T where T: LocalTrait {}
//~^ ERROR type parameter `T` must be used as the type parameter for some local type

fn main() {}
