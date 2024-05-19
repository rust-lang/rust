//@ check-pass

// Private types and traits are not allowed in public interfaces.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults)]

mod types {
    struct Priv;
    pub struct Pub;
    pub trait PubTr {
        type Alias;
    }

    pub const C: Priv = Priv; //~ WARNING type `types::Priv` is more private than the item `C`
    pub static S: Priv = Priv; //~ WARNING type `types::Priv` is more private than the item `S`
    pub fn f1(arg: Priv) {} //~ WARNING `types::Priv` is more private than the item `types::f1`
    pub fn f2() -> Priv { panic!() } //~ WARNING type `types::Priv` is more private than the item `types::f2`
    pub struct S1(pub Priv); //~ WARNING type `types::Priv` is more private than the item `types::S1::0`
    pub struct S2 { pub field: Priv } //~ WARNING `types::Priv` is more private than the item `S2::field`
    impl Pub {
        pub const C: Priv = Priv; //~ WARNING type `types::Priv` is more private than the item `types::Pub::C`
        pub fn f1(arg: Priv) {} //~ WARNING type `types::Priv` is more private than the item `types::Pub::f1`
        pub fn f2() -> Priv { panic!() } //~ WARNING type `types::Priv` is more private than the item `types::Pub::f2`
    }
}

mod traits {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub enum E<T: PrivTr> { V(T) } //~ WARNING trait `traits::PrivTr` is more private than the item `traits::E`
    pub fn f<T: PrivTr>(arg: T) {} //~ WARNING trait `traits::PrivTr` is more private than the item `traits::f`
    pub struct S1<T: PrivTr>(T); //~ WARNING trait `traits::PrivTr` is more private than the item `traits::S1`
    impl<T: PrivTr> Pub<T> { //~ WARNING trait `traits::PrivTr` is more private than the item `traits::Pub<T>`
        pub fn f<U: PrivTr>(arg: U) {} //~ WARNING trait `traits::PrivTr` is more private than the item `traits::Pub::<T>::f`
    }
}

mod traits_where {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub enum E<T> where T: PrivTr { V(T) }
    //~^ WARNING trait `traits_where::PrivTr` is more private than the item `traits_where::E`
    pub fn f<T>(arg: T) where T: PrivTr {}
    //~^ WARNING trait `traits_where::PrivTr` is more private than the item `traits_where::f`
    pub struct S1<T>(T) where T: PrivTr;
    //~^ WARNING trait `traits_where::PrivTr` is more private than the item `traits_where::S1`
    impl<T> Pub<T> where T: PrivTr {
    //~^ WARNING trait `traits_where::PrivTr` is more private than the item `traits_where::Pub<T>`
        pub fn f<U>(arg: U) where U: PrivTr {}
        //~^ WARNING trait `traits_where::PrivTr` is more private than the item `traits_where::Pub::<T>::f`
    }
}

mod generics {
    struct Priv<T = u8>(T);
    pub struct Pub<T = u8>(T);
    trait PrivTr<T> {}
    pub trait PubTr<T> {}

    pub fn f1(arg: [Priv; 1]) {} //~ WARNING type `generics::Priv` is more private than the item `generics::f1`
    pub fn f2(arg: Pub<Priv>) {} //~ WARNING type `generics::Priv` is more private than the item `generics::f2`
    pub fn f3(arg: Priv<Pub>) {}
    //~^ WARNING type `generics::Priv<generics::Pub>` is more private than the item `generics::f3`
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

    impl Pub {
        pub fn f(arg: Priv) {} //~ WARNING type `impls::Priv` is more private than the item `impls::Pub::f`
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
        type Assoc = m::Pub3;
    }
    impl PrivTr for Priv {}

    // This should be OK, but associated type aliases are not substituted yet
    pub fn f3(arg: <Priv as PrivTr>::Assoc) {}
    //~^ WARNING type `aliases_pub::Priv` is more private than the item `aliases_pub::f3`
    //~| WARNING associated type `aliases_pub::PrivTr::Assoc` is more private than the item `aliases_pub::f3`
    //~^^^ WARNING trait `aliases_pub::PrivTr` is more private than the item `aliases_pub::f3`

    impl PrivUseAlias {
        pub fn f(arg: Priv) {}
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
        type Assoc = Priv3;
    }
    impl PrivTr for Priv {}

    pub fn f1(arg: PrivUseAlias) {} //~ WARNING type `Priv1` is more private than the item `aliases_priv::f1`
    pub fn f2(arg: PrivAlias) {} //~ WARNING type `Priv2` is more private than the item `aliases_priv::f2`
    pub fn f3(arg: <Priv as PrivTr>::Assoc) {}
    //~^ WARNING type `aliases_priv::Priv` is more private than the item `aliases_priv::f3`
    //~| WARNING associated type `aliases_priv::PrivTr::Assoc` is more private than the item `aliases_priv::f3`
    //~^^^ WARNING trait `aliases_priv::PrivTr` is more private than the item `aliases_priv::f3`
}

mod aliases_params {
    struct Priv;
    type PrivAliasGeneric<T = Priv> = T;
    type Result<T> = ::std::result::Result<T, Priv>;

    pub fn f2(arg: PrivAliasGeneric) {}
    //~^ WARNING type `aliases_params::Priv` is more private than the item `aliases_params::f2`
    pub fn f3(arg: Result<u8>) {} //~ WARNING type `aliases_params::Priv` is more private than the item `aliases_params::f3`
}

fn main() {}
