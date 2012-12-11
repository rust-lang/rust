// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub mod num {
    pub trait Num2 {
        static pure fn from_int2(n: int) -> self;
    }
}

pub mod float {
    impl float: num::Num2 {
        #[inline]
        static pure fn from_int2(n: int) -> float { return n as float;  }
    }
}

