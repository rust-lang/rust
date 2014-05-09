// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations and constants for signed 8-bits integers (`i8` type)

use from_str::FromStr;
use num::{ToStrRadix, FromStrRadix};
use num::strconv;
use option::Option;
use slice::ImmutableVector;
use str;

pub use core::i8::{BITS, BYTES, MIN, MAX};

int_module!(i8)
