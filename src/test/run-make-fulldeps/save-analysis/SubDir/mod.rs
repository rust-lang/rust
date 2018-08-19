// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// sub-module in a sub-directory

use sub::sub2 as msalias;
use sub::sub2;

static yy: usize = 25;

mod sub {
    pub mod sub2 {
        pub mod sub3 {
            pub fn hello() {
                println!("hello from module 3");
            }
        }
        pub fn hello() {
            println!("hello from a module");
        }

        pub struct nested_struct {
            pub field2: u32,
        }
    }
}

pub struct SubStruct {
    pub name: String
}
