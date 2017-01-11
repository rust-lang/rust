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

#![feature(use_extern_macros)]

extern crate two_macros;

mod foo {
    pub mod bar {
        pub use two_macros::m;
    }
}

fn f() {
    use foo::*; //~ NOTE could also refer to the name imported here
    bar::m! { //~ ERROR ambiguous
              //~| NOTE macro-expanded items do not shadow when used in a macro invocation path
        mod bar { pub use two_macros::m; } //~ NOTE could refer to the name defined here
                                           //~^^^ NOTE in this expansion
    }
}

pub mod baz { //~ NOTE could also refer to the name defined here
    pub use two_macros::m;
}

fn g() {
    baz::m! { //~ ERROR ambiguous
              //~| NOTE macro-expanded items do not shadow when used in a macro invocation path
        mod baz { pub use two_macros::m; } //~ NOTE could refer to the name defined here
                                           //~^^^ NOTE in this expansion
    }
}
