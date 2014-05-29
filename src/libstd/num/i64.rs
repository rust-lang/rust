// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for signed 64-bits integers (`i64` type)

#![doc(primitive = "i64")]

use from_str::FromStr;
use num::{ToStrRadix, FromStrRadix};
use num::strconv;
use option::Option;
use slice::ImmutableVector;
use string::String;

pub use core::i64::{BITS, BYTES, MIN, MAX};

int_module!(i64)
