// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

// This ia file with many multi-byte characters, to try to encourage
// the compiler to trip on them.  The Drop implementation below will
// need to have its source location embedded into the debug info for
// the output file.

// αααααααααααααααααααααααααααααααααααααααααααααααααααααααααααααααααααααα
// ββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββββ
// γγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγγ
// δδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδδ
// εεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεεε

// ζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζζ
// ηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηηη
// θθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθθ
// ιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιιι
// κκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκκ

pub struct D<X:fmt::Debug>(pub X);

impl<X:fmt::Debug> Drop for D<X> {
    fn drop(&mut self) {
        // λλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλλ
        println!("Dropping D({:?})", self.0);
    }
}
