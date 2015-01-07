// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for pointer-sized unsigned integers (`usize` type)
//!
//! This type was recently added to replace `uint`. The rollout of the
//! new type will gradually take place over the alpha cycle along with
//! the development of clearer conventions around integer types.

#![stable]
#![doc(primitive = "usize")]

pub use core::usize::{BITS, BYTES, MIN, MAX};

uint_module! { usize }
