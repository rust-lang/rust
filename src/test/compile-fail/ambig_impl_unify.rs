// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait foo {
    fn foo() -> int;
}

impl ~[uint]: foo {
    fn foo() -> int {1} //~ NOTE candidate #1 is `__extensions__::foo`
}

impl ~[int]: foo {
    fn foo() -> int {2} //~ NOTE candidate #2 is `__extensions__::foo`
}

fn main() {
    let x = ~[];
    x.foo(); //~ ERROR multiple applicable methods in scope
}
