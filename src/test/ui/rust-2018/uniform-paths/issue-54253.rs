// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// edition:2018

// Dummy import that previously introduced uniform path canaries.
use std;

// fn version() -> &'static str {""}

mod foo {
    // Error wasn't reported, despite `version` being commented out above.
    use crate::version; //~ ERROR unresolved import `crate::version`

    fn bar() {
        version();
    }
}

fn main() {}
