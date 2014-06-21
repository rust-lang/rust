// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g. if `Add` changes to `Addition`, and
// `no_implicit_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

#[no_implicit_prelude]
mod foo {
    mod baz {
        struct Test;
        impl Add for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Clone for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Iterator for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl ToString for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Writer for Test {} //~ ERROR: attempt to implement a nonexistent trait

        fn foo() {
            drop(2) //~ ERROR: unresolved name
        }
    }

    struct Test;
    impl Add for Test {} //~ ERROR: attempt to implement a nonexistent trait
    impl Clone for Test {} //~ ERROR: attempt to implement a nonexistent trait
    impl Iterator for Test {} //~ ERROR: attempt to implement a nonexistent trait
    impl ToString for Test {} //~ ERROR: attempt to implement a nonexistent trait
    impl Writer for Test {} //~ ERROR: attempt to implement a nonexistent trait

    fn foo() {
        drop(2) //~ ERROR: unresolved name
    }
}

fn qux() {
    #[no_implicit_prelude]
    mod qux_inner {
        struct Test;
        impl Add for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Clone for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Iterator for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl ToString for Test {} //~ ERROR: attempt to implement a nonexistent trait
        impl Writer for Test {} //~ ERROR: attempt to implement a nonexistent trait

        fn foo() {
            drop(2) //~ ERROR: unresolved name
        }
    }
}


fn main() {
    // these should work fine
    drop(2)
}
