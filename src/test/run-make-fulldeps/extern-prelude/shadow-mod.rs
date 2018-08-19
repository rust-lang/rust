// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Local module shadows `ep_lib` from extern prelude

mod ep_lib {
    pub struct S;

    impl S {
        pub fn internal(&self) {}
    }
}

fn main() {
    let s = ep_lib::S;
    s.internal(); // OK
}
