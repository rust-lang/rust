// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a supertrait cycle where the first trait we find (`A`) is not
// a direct participant in the cycle.

trait A: B {
    //~^ NOTE the cycle begins when computing the supertraits of `B`...
}

trait B: C {
    //~^ NOTE ...which then requires computing the supertraits of `C`...
}

trait C: B { }
    //~^ ERROR unsupported cyclic reference
    //~| cyclic reference
    //~| NOTE ...which then again requires computing the supertraits of `B`, completing the cycle

fn main() { }
