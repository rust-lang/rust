// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See comments in Cargo.toml for why this exists

// There's a bug right now where if we pass --extern std=... and we're cross
// compiling then this doesn't work with `#[macro_use] extern crate std;`. Work
// around this by not having `#[macro_use] extern crate std;`
#![no_std]
extern crate std;
