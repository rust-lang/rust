// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Previously, this would have failed to resolve due to the circular
// block between `use say` and `pub use hello::*`.
//
// Now, as `use say` is not `pub`, the glob import can resolve
// without any problem and this resolves fine.

pub use hello::*;

pub mod say {
    pub fn hello() { println!("hello"); }
}

pub mod hello {
    use say;

    pub fn hello() {
        say::hello();
    }
}

fn main() {
    hello();
}
