// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:two_macros.rs

#![feature(item_like_imports, use_extern_macros)]

extern crate two_macros; // two identity macros `m` and `n`

mod foo {
    pub use two_macros::n as m;
}

mod m1 {
    m!(use two_macros::*;);
    use foo::m; // This shadows the glob import
}

mod m2 {
    use two_macros::*; //~ NOTE could also refer
    m! { //~ ERROR ambiguous
         //~| NOTE macro-expanded macro imports do not shadow
        use foo::m; //~ NOTE could refer to the name imported here
                    //~^^^ NOTE in this expansion
    }
}

mod m3 {
    use two_macros::m; //~ NOTE could also refer
    fn f() {
        use two_macros::n as m; // This shadows the above import
        m!();
    }

    fn g() {
        m! { //~ ERROR ambiguous
             //~| NOTE macro-expanded macro imports do not shadow
            use two_macros::n as m; //~ NOTE could refer to the name imported here
                                    //~^^^ NOTE in this expansion
        }
    }
}

mod m4 {
    macro_rules! m { () => {} } //~ NOTE could refer to the macro defined here
    use two_macros::m; //~ NOTE could also refer to the macro imported here
    m!(); //~ ERROR ambiguous
}
