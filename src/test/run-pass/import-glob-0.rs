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
    #[legacy_exports];
    export f1;
    export f2;
    export f4;
    fn f1() { debug!("f1"); }
    fn f2() { debug!("f2"); }
    fn f3() { debug!("f3"); }
    fn f4() { debug!("f4"); }
}

mod dug {
    #[legacy_exports];
    mod too {
        #[legacy_exports];
        mod greedily {
            #[legacy_exports];
            mod and {
                #[legacy_exports];
                mod too {
                    #[legacy_exports];
                    mod deep {
                        #[legacy_exports];
                        fn nameless_fear() { debug!("Boo!"); }
                        fn also_redstone() { debug!("Whatever."); }
                    }
                }
            }
        }
    }
}


fn main() { f1(); f2(); f4(); nameless_fear(); also_redstone(); }
