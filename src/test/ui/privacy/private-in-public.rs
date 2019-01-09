// Private types and traits are not allowed in public interfaces.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults)]

mod types {
    struct Priv;
    pub struct Pub;
    pub trait PubTr {
        type Alias;
    }

    pub const C: Priv = Priv; //~ ERROR private type `types::Priv` in public interface
    pub static S: Priv = Priv; //~ ERROR private type `types::Priv` in public interface
    pub fn f1(arg: Priv) {} //~ ERROR private type `types::Priv` in public interface
    pub fn f2() -> Priv { panic!() } //~ ERROR private type `types::Priv` in public interface
    pub struct S1(pub Priv); //~ ERROR private type `types::Priv` in public interface
    pub struct S2 { pub field: Priv } //~ ERROR private type `types::Priv` in public interface
    impl Pub {
        pub const C: Priv = Priv; //~ ERROR private type `types::Priv` in public interface
        pub fn f1(arg: Priv) {} //~ ERROR private type `types::Priv` in public interface
        pub fn f2() -> Priv { panic!() } //~ ERROR private type `types::Priv` in public interface
    }
}

mod traits {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub enum E<T: PrivTr> { V(T) } //~ ERROR private trait `traits::PrivTr` in public interface
    pub fn f<T: PrivTr>(arg: T) {} //~ ERROR private trait `traits::PrivTr` in public interface
    pub struct S1<T: PrivTr>(T); //~ ERROR private trait `traits::PrivTr` in public interface
    impl<T: PrivTr> Pub<T> { //~ ERROR private trait `traits::PrivTr` in public interface
        pub fn f<U: PrivTr>(arg: U) {} //~ ERROR private trait `traits::PrivTr` in public interface
    }
}

mod traits_where {
    trait PrivTr {}
    pub struct Pub<T>(T);
    pub trait PubTr {}

    pub enum E<T> where T: PrivTr { V(T) }
    //~^ ERROR private trait `traits_where::PrivTr` in public interface
    pub fn f<T>(arg: T) where T: PrivTr {}
    //~^ ERROR private trait `traits_where::PrivTr` in public interface
    pub struct S1<T>(T) where T: PrivTr;
    //~^ ERROR private trait `traits_where::PrivTr` in public interface
    impl<T> Pub<T> where T: PrivTr {
    //~^ ERROR private trait `traits_where::PrivTr` in public interface
        pub fn f<U>(arg: U) where U: PrivTr {}
        //~^ ERROR private trait `traits_where::PrivTr` in public interface
    }
}

mod generics {
    struct Priv<T = u8>(T);
    pub struct Pub<T = u8>(T);
    trait PrivTr<T> {}
    pub trait PubTr<T> {}

    pub fn f1(arg: [Priv; 1]) {} //~ ERROR private type `generics::Priv` in public interface
    pub fn f2(arg: Pub<Priv>) {} //~ ERROR private type `generics::Priv` in public interface
    pub fn f3(arg: Priv<Pub>) {}
    //~^ ERROR private type `generics::Priv<generics::Pub>` in public interface
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
        pub fn f(arg: Priv) {} //~ ERROR private type `impls::Priv` in public interface
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
    //~^ ERROR private trait `aliases_pub::PrivTr` in public interface
    //~| ERROR private type `aliases_pub::Priv` in public interface

    impl PrivUseAlias {
        pub fn f(arg: Priv) {} //~ ERROR private type `aliases_pub::Priv` in public interface
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

    pub fn f1(arg: PrivUseAlias) {} //~ ERROR private type `aliases_priv::Priv1` in public interface
    pub fn f2(arg: PrivAlias) {} //~ ERROR private type `aliases_priv::Priv2` in public interface
    pub fn f3(arg: <Priv as PrivTr>::Assoc) {}
    //~^ ERROR private trait `aliases_priv::PrivTr` in public interface
    //~| ERROR private type `aliases_priv::Priv` in public interface
}

mod aliases_params {
    struct Priv;
    type PrivAliasGeneric<T = Priv> = T;
    type Result<T> = ::std::result::Result<T, Priv>;

    pub fn f2(arg: PrivAliasGeneric) {}
    //~^ ERROR private type `aliases_params::Priv` in public interface
    pub fn f3(arg: Result<u8>) {} //~ ERROR private type `aliases_params::Priv` in public interface
}

fn main() {}
