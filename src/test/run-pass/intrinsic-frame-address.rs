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
#[legacy_modes];

mod rusti {
    #[abi = "rust-intrinsic"]
    pub extern {
        pub fn frame_address(f: &once fn(*u8));
    }
}

pub fn main() {
    unsafe {
        do rusti::frame_address |addr| {
            assert!(addr.is_not_null());
        }
    }
}
