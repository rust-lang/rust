// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for architecture-sized unsigned integers (`uint` type)

#![unstable]
#![doc(primitive = "uint")]

use from_str::FromStr;
use num::{ToStrRadix, FromStrRadix};
use num::strconv;
use option::Option;
use slice::ImmutableVector;
use string::String;

pub use core::uint::{BITS, BYTES, MIN, MAX};

uint_module!(uint)
