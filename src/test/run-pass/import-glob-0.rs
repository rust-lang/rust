// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use module_of_many_things::*;
use dug::too::greedily::and::too::deep::*;

mod module_of_many_things {
    pub fn f1() { info!("f1"); }
    pub fn f2() { info!("f2"); }
    fn f3() { info!("f3"); }
    pub fn f4() { info!("f4"); }
}

mod dug {
    pub mod too {
        pub mod greedily {
            pub mod and {
                pub mod too {
                    pub mod deep {
                        pub fn nameless_fear() { info!("Boo!"); }
                        pub fn also_redstone() { info!("Whatever."); }
                    }
                }
            }
        }
    }
}


pub fn main() { f1(); f2(); f4(); nameless_fear(); also_redstone(); }
