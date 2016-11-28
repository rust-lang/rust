// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod m1 {
    pub struct Pub;
    struct Priv;

    impl Pub {
        pub fn f() -> Priv {Priv} //~ ERROR private type `m1::Priv` in public interface
    }
}

mod m2 {
    #![deny(future_incompatible)]

    pub struct Pub;
    struct Priv;

    impl Pub {
        pub fn f() -> Priv {Priv} //~ ERROR private type `m2::Priv` in public interface
    }
}

fn main() {}
