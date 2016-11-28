// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(private_in_public)]
#![allow(unused)]

struct SemiPriv;

mod m1 {
    struct Priv;
    impl ::SemiPriv {
        pub fn f(_: Priv) {} //~ ERROR private type `m1::Priv` in public interface
        //~^ WARNING hard error
    }

    impl Priv {
        pub fn f(_: Priv) {} // ok
    }
}

mod m2 {
    struct Priv;
    impl ::std::ops::Deref for ::SemiPriv {
        type Target = Priv; //~ ERROR private type `m2::Priv` in public interface
        //~^ WARNING hard error
        fn deref(&self) -> &Self::Target { unimplemented!() }
    }

    impl ::std::ops::Deref for Priv {
        type Target = Priv; // ok
        fn deref(&self) -> &Self::Target { unimplemented!() }
    }
}

trait SemiPrivTrait {
    type Assoc;
}

mod m3 {
    struct Priv;
    impl ::SemiPrivTrait for () {
        type Assoc = Priv; //~ ERROR private type `m3::Priv` in public interface
        //~^ WARNING hard error
    }
}

fn main() {}
