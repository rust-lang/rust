// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod aliases_pub {
    struct Priv;
    mod m {
        pub struct Pub3;
    }

    trait PrivTr {
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = m::Pub3;
    }

    impl <Priv as PrivTr>::AssocAlias { //~ ERROR no base type found for inherent implementation
        pub fn f(arg: Priv) {} // private type `aliases_pub::Priv` in public interface
    }
}

mod aliases_priv {
    struct Priv;
    struct Priv3;

    trait PrivTr {
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = Priv3;
    }

    impl <Priv as PrivTr>::AssocAlias { //~ ERROR no base type found for inherent implementation
        pub fn f(arg: Priv) {} // OK
    }
}

fn main() {}
