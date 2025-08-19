// Private types and traits are not allowed in public interfaces.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults)]
#![deny(private_interfaces, private_bounds)]
#![allow(improper_ctypes)]

mod types {
    struct Priv;
    pub struct Pub;
    pub trait PubTr {
        type Alias;
    }

    pub type Alias = Priv; //~ ERROR type `types::Priv` is more private than the item `types::Alias`
    pub enum E {
        V1(Priv), //~ ERROR type `types::Priv` is more private than the item `E::V1::0`
        V2 { field: Priv }, //~ ERROR type `types::Priv` is more private than the item `E::V2::field`
    }
    pub trait Tr {
        const C: Priv = Priv; //~ ERROR type `types::Priv` is more private than the item `Tr::C`
        type Alias = Priv; //~ ERROR private type `types::Priv` in public interface
        fn f1(arg: Priv) {} //~ ERROR type `types::Priv` is more private than the item `Tr::f1`
        fn f2() -> Priv { panic!() } //~ ERROR type `types::Priv` is more private than the item `Tr::f2`
    }
    extern "C" {
        pub static ES: Priv; //~ ERROR type `types::Priv` is more private than the item `types::ES`
        pub fn ef1(arg: Priv); //~ ERROR type `types::Priv` is more private than the item `types::ef1`
        pub fn ef2() -> Priv; //~ ERROR type `types::Priv` is more private than the item `types::ef2`
    }
    impl PubTr for Pub {
        type Alias = Priv; //~ ERROR private type `types::Priv` in public interface
    }
}

mod traits {
    trait PrivTr {}
    impl PrivTr for () {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub type Alias<T: PrivTr> = T; //~ ERROR trait `traits::PrivTr` is more private than the item `traits::Alias`
    //~^ WARNING bounds on generic parameters in type aliases are not enforced
    pub trait Tr1: PrivTr {} //~ ERROR trait `traits::PrivTr` is more private than the item `traits::Tr1`
    pub trait Tr2<T: PrivTr> {} //~ ERROR trait `traits::PrivTr` is more private than the item `traits::Tr2`
    pub trait Tr3 {
        type Alias: PrivTr;
        //~^ ERROR trait `traits::PrivTr` is more private than the item `traits::Tr3::Alias`
        fn f<T: PrivTr>(arg: T) {}
        //~^ ERROR trait `traits::PrivTr` is more private than the item `traits::Tr3::f`
        fn g() -> impl PrivTr;
        fn h() -> impl PrivTr {}
    }
    impl<T: PrivTr> Pub<T> {} //~ ERROR trait `traits::PrivTr` is more private than the item `traits::Pub<T>`
    impl<T: PrivTr> PubTr for Pub<T> {} // OK, trait impl predicates
}

mod traits_where {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub type Alias<T> where T: PrivTr = T;
        //~^ ERROR trait `traits_where::PrivTr` is more private than the item `traits_where::Alias`
        //~| WARNING where clauses on type aliases are not enforced
    pub trait Tr2<T> where T: PrivTr {}
        //~^ ERROR trait `traits_where::PrivTr` is more private than the item `traits_where::Tr2`
    pub trait Tr3 {
        fn f<T>(arg: T) where T: PrivTr {}
        //~^ ERROR trait `traits_where::PrivTr` is more private than the item `traits_where::Tr3::f`
    }
    impl<T> Pub<T> where T: PrivTr {}
        //~^ ERROR trait `traits_where::PrivTr` is more private than the item `traits_where::Pub<T>`
    impl<T> PubTr for Pub<T> where T: PrivTr {} // OK, trait impl predicates
}

mod generics {
    struct Priv<T = u8>(T);
    pub struct Pub<T = u8>(T);
    trait PrivTr<T> {}
    pub trait PubTr<T> {}
    impl PrivTr<Priv<()>> for () {}

    pub trait Tr1: PrivTr<Pub> {}
        //~^ ERROR trait `generics::PrivTr<generics::Pub>` is more private than the item `generics::Tr1`
    pub trait Tr2: PubTr<Priv> {} //~ ERROR type `generics::Priv` is more private than the item `generics::Tr2`
    pub trait Tr3: PubTr<[Priv; 1]> {} //~ ERROR type `generics::Priv` is more private than the item `generics::Tr3`
    pub trait Tr4: PubTr<Pub<Priv>> {} //~ ERROR type `generics::Priv` is more private than the item `Tr4`

    pub trait Tr5 {
        fn required() -> impl PrivTr<Priv<()>>;
        fn provided() -> impl PrivTr<Priv<()>> {}
    }
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
        type Alias = Priv; //~ ERROR private type `impls::Priv` in public interface
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
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = m::Pub3;
    }

    pub fn f1(arg: PrivUseAlias) {} // OK
    pub fn f2(arg: PrivAlias) {} // OK

    pub trait Tr1: PrivUseAliasTr {} // OK
    pub trait Tr2: PrivUseAliasTr<PrivAlias> {} // OK

    impl PrivAlias {
        pub fn f(arg: Priv) {} //~ ERROR type `aliases_pub::Priv` is more private than the item `aliases_pub::<impl Pub2>::f`
    }
    impl PrivUseAliasTr for PrivUseAlias {
        type Check = Priv; //~ ERROR private type `aliases_pub::Priv` in public interface
    }
    impl PrivUseAliasTr for PrivAlias {
        type Check = Priv; //~ ERROR private type `aliases_pub::Priv` in public interface
    }
    impl PrivUseAliasTr for <Priv as PrivTr>::AssocAlias {
        type Check = Priv; //~ ERROR private type `aliases_pub::Priv` in public interface
    }
    impl PrivUseAliasTr for Option<<Priv as PrivTr>::AssocAlias> {
        type Check = Priv; //~ ERROR private type `aliases_pub::Priv` in public interface
    }
    impl PrivUseAliasTr for (<Priv as PrivTr>::AssocAlias, Priv) {
        type Check = Priv; // OK
    }
    impl PrivUseAliasTr for Option<(<Priv as PrivTr>::AssocAlias, Priv)> {
        type Check = Priv; // OK
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
        type AssocAlias;
    }
    impl PrivTr for Priv {
        type AssocAlias = Priv3;
    }

    pub trait Tr1: PrivUseAliasTr {}
        //~^ ERROR trait `PrivTr1` is more private than the item `aliases_priv::Tr1`
    pub trait Tr2: PrivUseAliasTr<PrivAlias> {}
        //~^ ERROR trait `PrivTr1<Priv2>` is more private than the item `aliases_priv::Tr2`
        //~| ERROR type `Priv2` is more private than the item `aliases_priv::Tr2`

    impl PrivUseAlias {
        pub fn f(arg: Priv) {} // OK
    }
    impl PrivAlias {
        pub fn f(arg: Priv) {} // OK
    }
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

    pub fn f1(arg: PrivAliasGeneric<u8>) {} // OK, not an error
}

fn main() {}
