// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate extra;

trait SipHash {
    fn reset(&self);
}

fn siphash(k0 : u64) -> SipHash {
    struct SipState {
        v0: u64,
    }

    impl SipHash for SipState {
        fn reset(&self) {
           self.v0 = k0 ^ 0x736f6d6570736575; //~ ERROR can't capture dynamic environment
           //~^ ERROR unresolved name `k0`.
        }
    }
    fail!();
}

fn main() {}
