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

mod a {
    #[legacy_exports];
    mod b {
        #[legacy_exports];
        mod a {
            #[legacy_exports];
            fn foo() -> int { return 1; }
        }
    }
}

mod c {
    #[legacy_exports];
    use a::b::a;
    fn bar() { assert (a::foo() == 1); }
}

fn main() { c::bar(); }
