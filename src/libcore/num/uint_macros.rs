// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]
#![doc(hidden)]

macro_rules! uint_module (($T:ty, $T_SIGNED:ty, $bits:expr) => (

#[unstable]
pub const BITS : uint = $bits;
#[unstable]
pub const BYTES : uint = ($bits / 8);

#[unstable]
pub const MIN: $T = 0 as $T;
#[unstable]
pub const MAX: $T = 0 as $T - 1 as $T;

))
