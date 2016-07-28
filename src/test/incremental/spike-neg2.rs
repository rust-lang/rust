// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A variant of the first "spike" test that serves to test the
// `rustc_partition_reused` and `rustc_partition_translated` tests.
// Here we change and say that the `y` module will be translated (when
// in fact it will not), and then indicate that the test itself
// should-fail (because an error will be reported, and hence the
// revision rpass2 will not compile, despite being named rpass).

// revisions:rpass1 rpass2
// should-fail

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="spike_neg2", cfg="rpass2")]
#![rustc_partition_translated(module="spike_neg2-x", cfg="rpass2")]
#![rustc_partition_translated(module="spike_neg2-y", cfg="rpass2")] // this is wrong!

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
