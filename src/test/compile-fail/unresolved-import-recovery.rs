// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that unresolved imports do not create additional errors and ICEs

mod m {
    pub use unresolved; //~ ERROR unresolved import `unresolved`

    fn f() {
        let unresolved = 0; // OK
    }
}

fn main() {
    match 0u8 {
        m::unresolved => {} // OK
        m::unresolved(..) => {} // OK
        m::unresolved{..} => {} // OK
    }
}
