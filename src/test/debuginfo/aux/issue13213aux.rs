// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]
// compile-flags:-g

pub use private::P;

#[derive(Copy, Clone)]
pub struct S {
    p: P,
}

mod private {
    #[derive(Copy, Clone)]
    pub struct P {
        p: i32,
    }
    pub const THREE: P = P { p: 3 };
}

pub static A: S = S { p: private::THREE };
