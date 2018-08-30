// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the self param space is not used in a conflicting
// manner by unboxed closures within a default method on a trait

// pretty-expanded FIXME #23616

trait Tr {
    fn foo(&self);

    fn bar(&self) {
        (|| { self.foo() })()
    }
}

impl Tr for () {
    fn foo(&self) {}
}

fn main() {
    ().bar();
}
