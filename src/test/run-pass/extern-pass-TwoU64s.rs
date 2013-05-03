// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a foreign function that accepts and returns a struct
// by value.

// xfail-fast This works standalone on windows but not with check-fast.
// possibly because there is another test that uses this extern fn but gives it
// a diferent signature

#[deriving(Eq)]
struct TwoU64s {
    one: u64, two: u64
}

pub extern {
    pub fn rust_dbg_extern_identity_TwoU64s(v: TwoU64s) -> TwoU64s;
}

pub fn main() {
    unsafe {
        let x = TwoU64s {one: 22, two: 23};
        let y = rust_dbg_extern_identity_TwoU64s(x);
        assert!(x == y);
    }
}
