// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:xc_conditions_4.rs

extern mod xc_conditions_4;
use xcc = xc_conditions_4;

struct SThunk {
    x: int
}

impl xcc::Thunk<xcc::Color> for SThunk {
    fn call(self) -> xcc::Color {
        xcc::oops::cond.raise((self.x, 1.23, ~"oh no"))
    }
}

pub fn main() {
    do xcc::oops::cond.trap(|_| xcc::Red).inside {
        let t = SThunk { x : 10 };
        assert_eq!(xcc::callback(t), xcc::Red)
    }
}