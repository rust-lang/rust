// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct TwoU32s {
    one: u32, two: u32
}

extern {
    pub fn rust_dbg_extern_return_TwoU32s() -> TwoU32s;
}

pub fn main() {
    unsafe {
        let y = rust_dbg_extern_return_TwoU32s();
        assert_eq!(y.one, 10);
        assert_eq!(y.two, 20);
    }
}
