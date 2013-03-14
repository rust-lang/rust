// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod rusti {
    #[abi = "rust-intrinsic"]
    pub extern "rust-intrinsic" {
        pub fn move_val_init<T>(dst: &mut T, +src: T);
        pub fn move_val<T>(dst: &mut T, +src: T);
    }
}

pub fn main() {
    unsafe {
        let mut x = @1;
        let mut y = @2;
        rusti::move_val(&mut y, x);
        assert!(*y == 1);
    }
}
