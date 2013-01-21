// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use prelude::Copy;

/// Clonable types are copied with the clone method
pub trait Clone {
    fn clone(&self) -> self;
}

impl <T: Copy> T: Clone {
    #[inline(always)]
    fn clone(&self) -> T { copy *self }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_int() {
        let x = 5;
        let y = x.clone();
        assert x == y;
    }

    #[test]
    fn test_str() {
        let x = ~"foo";
        let y = x.clone();
        assert x == y;
    }
}
