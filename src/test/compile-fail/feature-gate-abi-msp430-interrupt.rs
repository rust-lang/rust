// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the MSP430 interrupt ABI cannot be used when msp430_interrupt
// feature gate is not used.

extern "msp430-interrupt" fn foo() {}
//~^ ERROR msp430-interrupt ABI is experimental and subject to change

fn main() {
    foo();
}
