// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A first "spike" for incremental compilation: here, we change the
// content of the `make` function, and we find that we can reuse the
// `y` module entirely (but not the `x` module).

// revisions:rpass1 rpass2

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="spike", cfg="rpass2")]
#![rustc_partition_translated(module="spike-x", cfg="rpass2")]
#![rustc_partition_reused(module="spike-y", cfg="rpass2")]

mod x {
    pub struct X {
        x: u32, y: u32,
    }

    #[cfg(rpass1)]
    fn make() -> X {
        X { x: 22, y: 0 }
    }

    #[cfg(rpass2)]
    fn make() -> X {
        X { x: 11, y: 11 }
    }

    pub fn new() -> X {
        make()
    }

    pub fn sum(x: &X) -> u32 {
        x.x + x.y
    }
}

mod y {
    use x;

    pub fn assert_sum() -> bool {
        let x = x::new();
        x::sum(&x) == 22
    }
}

pub fn main() {
    y::assert_sum();
}
