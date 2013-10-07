// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// regression test for issue 4366

// ensures that 'use foo:*' doesn't import non-public 'use' statements in the
// module 'foo'

#[feature(globs)];

use m1::*;

mod foo {
    pub fn foo() {}
}
mod a {
    pub mod b {
        use foo::foo;
        type bar = int;
    }
    pub mod sub {
        use a::b::*;
        fn sub() -> bar { foo(); 1 } //~ ERROR: unresolved name `foo`
        //~^ ERROR: use of undeclared type name `bar`
    }
}

mod m1 {
    fn foo() {}
}

fn main() {
    foo(); //~ ERROR: unresolved name `foo`
}
