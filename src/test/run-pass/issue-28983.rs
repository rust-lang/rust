// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Test { type T; }

impl Test for u32 {
    type T = i32;
}

pub mod export {
    #[no_mangle]
    pub extern "C" fn issue_28983(t: <u32 as ::Test>::T) -> i32 { t*3 }
}

// to test both exporting and importing functions, import
// a function from ourselves.
extern "C" {
    fn issue_28983(t: <u32 as Test>::T) -> i32;
}

fn main() {
    assert_eq!(export::issue_28983(2), 6);
    assert_eq!(unsafe { issue_28983(3) }, 9);
}
