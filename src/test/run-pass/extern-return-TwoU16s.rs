// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-win32 #5745

struct TwoU16s {
    one: u16, two: u16
}

pub extern {
    pub fn rust_dbg_extern_return_TwoU16s() -> TwoU16s;
}

pub fn main() {
    unsafe {
        let y = rust_dbg_extern_return_TwoU16s();
        assert!(y.one == 10);
        assert!(y.two == 20);
    }
}
