// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Private types and traits are not allowed in public interfaces.
// This test also ensures that the checks are performed even inside private modules.

#![feature(rustc_attrs)]
#![feature(associated_consts)]
#![feature(associated_type_defaults)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(improper_ctypes)]

mod types {
    struct Priv;
    pub struct Pub;
    pub trait PubTr {
        type Alias;
    }

    pub type Alias = Priv; //~ WARN private type in public interface
    //~^ WARNING HARD ERROR
    pub enum E {
        V1(Priv), //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        V2 { field: Priv }, //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    pub trait Tr {
        const C: Priv = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        type Alias = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        fn f1(arg: Priv) {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        fn f2() -> Priv { panic!() } //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    extern {
        pub static ES: Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        pub fn ef1(arg: Priv); //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
        pub fn ef2() -> Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    impl PubTr for Pub {
        type Alias = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
}

mod traits {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub type Alias<T: PrivTr> = T; //~ WARN private trait in public interface
    //~^ WARN trait bounds are not (yet) enforced in type definitions
    //~| WARNING HARD ERROR
    pub trait Tr1: PrivTr {} //~ WARN private trait in public interface
    //~^ WARNING HARD ERROR
    pub trait Tr2<T: PrivTr> {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr3 {
        type Alias: PrivTr; //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
        fn f<T: PrivTr>(arg: T) {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    }
    impl<T: PrivTr> Pub<T> {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    impl<T: PrivTr> PubTr for Pub<T> {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
}

mod traits_where {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub type Alias<T> where T: PrivTr = T; //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr2<T> where T: PrivTr {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr3 {
        fn f<T>(arg: T) where T: PrivTr {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    }
    impl<T> Pub<T> where T: PrivTr {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    impl<T> PubTr for Pub<T> where T: PrivTr {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
}

mod generics {
    struct Priv<T = u8>(T);
    pub struct Pub<T = u8>(T);
    trait PrivTr<T> {}
    pub trait PubTr<T> {}

    pub trait Tr1: PrivTr<Pub> {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr2: PubTr<Priv> {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr3: PubTr<[Priv; 1]> {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr4: PubTr<Pub<Priv>> {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
}

mod impls {
    struct Priv;
    pub struct Pub;
    trait PrivTr {
        type Alias;
    }
    pub trait PubTr {
        type Alias;
    }

    impl Priv {
        pub fn f(arg: Priv) {} // OK
    }
    impl PrivTr for Priv {
        type Alias = Priv; // OK
    }
    impl PubTr for Priv {
        type Alias = Priv; // OK
    }
    impl PrivTr for Pub {
        type Alias = Priv; // OK
    }
    impl PubTr for Pub {
        type Alias = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
}

mod impls_generics {
    struct Priv<T = u8>(T);
    pub struct Pub<T = u8>(T);
    trait PrivTr<T = u8> {
        type Alias;
    }
    pub trait PubTr<T = u8> {
        type Alias;
    }

    impl Priv<Pub> {
        pub fn f(arg: Priv) {} // OK
    }
    impl Pub<Priv> {
        pub fn f(arg: Priv) {} // OK
    }
    impl PrivTr<Pub> for Priv {
        type Alias = Priv; // OK
    }
    impl PubTr<Priv> for Priv {
        type Alias = Priv; // OK
    }
    impl PubTr for Priv<Pub> {
        type Alias = Priv; // OK
    }
    impl PubTr for [Priv; 1] {
        type Alias = Priv; // OK
    }
    impl PubTr for Pub<Priv> {
        type Alias = Priv; // OK
    }
    impl PrivTr<Pub> for Pub {
        type Alias = Priv; // OK
    }
    impl PubTr<Priv> for Pub {
        type Alias = Priv; // OK
    }
}

mod aliases_pub {
    struct Priv;
    mod m {
        pub struct Pub1;
        pub struct Pub2;
        pub struct Pub3;
        pub trait PubTr<T = u8> {
            type Check = u8;
        }
    }

    use self::m::Pub1 as PrivUseAlias;
    use self::m::PubTr as PrivUseAliasTr;
    type PrivAlias = m::Pub2;
    trait PrivTr {
        type AssocAlias = m::Pub3;
    }
    impl PrivTr for Priv {}

    pub fn f1(arg: PrivUseAlias) {} // OK

    pub trait Tr1: PrivUseAliasTr {} // OK
    // This should be OK, if type aliases are substituted
    pub trait Tr2: PrivUseAliasTr<PrivAlias> {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR

    impl PrivAlias {
        pub fn f(arg: Priv) {} //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    // This doesn't even parse
    // impl <Priv as PrivTr>::AssocAlias {
    //     pub fn f(arg: Priv) {} // WARN private type in public interface
    // }
    impl PrivUseAliasTr for PrivUseAlias {
        type Check = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    impl PrivUseAliasTr for PrivAlias {
        type Check = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
    impl PrivUseAliasTr for <Priv as PrivTr>::AssocAlias {
        type Check = Priv; //~ WARN private type in public interface
        //~^ WARNING HARD ERROR
    }
}

mod aliases_priv {
    struct Priv;

    struct Priv1;
    struct Priv2;
    struct Priv3;
    trait PrivTr1<T = u8> {
        type Check = u8;
    }

    use self::Priv1 as PrivUseAlias;
    use self::PrivTr1 as PrivUseAliasTr;
    type PrivAlias = Priv2;
    trait PrivTr {
        type AssocAlias = Priv3;
    }
    impl PrivTr for Priv {}

    pub trait Tr1: PrivUseAliasTr {} //~ WARN private trait in public interface
        //~^ WARNING HARD ERROR
    pub trait Tr2: PrivUseAliasTr<PrivAlias> {} //~ WARN private trait in public interface
     //~^ WARN private type in public interface
        //~| WARNING HARD ERROR
        //~| WARNING HARD ERROR

    impl PrivUseAlias {
        pub fn f(arg: Priv) {} // OK
    }
    impl PrivAlias {
        pub fn f(arg: Priv) {} // OK
    }
    // This doesn't even parse
    // impl <Priv as PrivTr>::AssocAlias {
    //     pub fn f(arg: Priv) {} // OK
    // }
    impl PrivUseAliasTr for PrivUseAlias {
        type Check = Priv; // OK
    }
    impl PrivUseAliasTr for PrivAlias {
        type Check = Priv; // OK
    }
    impl PrivUseAliasTr for <Priv as PrivTr>::AssocAlias {
        type Check = Priv; // OK
    }
}

mod aliases_params {
    struct Priv;
    type PrivAliasGeneric<T = Priv> = T;
    type Result<T> = ::std::result::Result<T, Priv>;
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
