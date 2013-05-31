// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

trait siphash {
    fn result(&self) -> u64;
    fn reset(&self);
}

fn siphash(k0 : u64, k1 : u64) -> siphash {
    struct SipState {
        v0: u64,
        v1: u64,
    }

    fn mk_result(st : SipState) -> u64 {

        let v0 = st.v0,
            v1 = st.v1;
        return v0 ^ v1;
    }

   impl siphash for SipState {
        fn reset(&self) {
            self.v0 = k0 ^ 0x736f6d6570736575;  //~ ERROR attempted dynamic environment-capture
            //~^ ERROR unresolved name `k0`.
            self.v1 = k1 ^ 0x646f72616e646f6d;   //~ ERROR attempted dynamic environment-capture
            //~^ ERROR unresolved name `k1`.
        }
        fn result(&self) -> u64 { return mk_result(self); }
    }
}

fn main() {}
