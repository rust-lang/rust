// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]
#![allow(unreachable_code)]

fn foo() { //~ ERROR function is never used

    // none of these should have any dead_code exposed to the user
    panic!();

    panic!("foo");

    panic!("bar {}", "baz")
}


fn main() {}
