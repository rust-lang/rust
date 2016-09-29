// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:a.rs
// aux-build:b.rs
// revisions:rpass1 rpass2

#![feature(rustc_attrs)]


#[cfg(rpass1)]
extern crate a;
#[cfg(rpass1)]
extern crate b;

#[cfg(rpass2)]
extern crate b;
#[cfg(rpass2)]
extern crate a;

use a::A;
use b::B;

//? #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
pub fn main() {
    A + B;
}
