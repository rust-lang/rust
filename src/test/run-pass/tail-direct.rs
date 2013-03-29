// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
pub fn main() { assert!((even(42))); assert!((odd(45))); }

fn even(n: int) -> bool { if n == 0 { return true; } else { return odd(n - 1); } }

fn odd(n: int) -> bool { if n == 0 { return false; } else { return even(n - 1); } }
