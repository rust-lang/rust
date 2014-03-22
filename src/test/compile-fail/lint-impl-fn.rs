// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(while_true)]
#![allow(dead_code)]

struct A(int);

impl A {
    fn foo(&self) { while true {} }

    #[deny(while_true)]
    fn bar(&self) { while true {} } //~ ERROR: infinite loops
}

#[deny(while_true)]
mod foo {
    struct B(int);

    impl B {
        fn foo(&self) { while true {} } //~ ERROR: infinite loops

        #[allow(while_true)]
        fn bar(&self) { while true {} }
    }
}

#[deny(while_true)]
fn main() {
    while true {} //~ ERROR: infinite loops
}
