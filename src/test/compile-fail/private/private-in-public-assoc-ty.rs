// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Private types and traits are not allowed in interfaces of associated types.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults)]

mod m {
    struct Priv;
    trait PrivTr {}
    impl PrivTr for Priv {}
    pub trait PubTrAux1<T> {}
    pub trait PubTrAux2 { type A; }

    // "Private-in-public in associated types is hard error" in RFC 2145
    // applies only to the aliased types, not bounds.
    pub trait PubTr {
        //~^ WARN private trait `m::PrivTr` in public interface
        //~| WARN this was previously accepted
        //~| WARN private type `m::Priv` in public interface
        //~| WARN this was previously accepted
        type Alias1: PrivTr;
        type Alias2: PubTrAux1<Priv> = u8;
        type Alias3: PubTrAux2<A = Priv> = u8;

        type Alias4 = Priv;
        //~^ ERROR private type `m::Priv` in public interface
    }
    impl PubTr for u8 {
        type Alias1 = Priv;
        //~^ ERROR private type `m::Priv` in public interface
    }
}

fn main() {}
