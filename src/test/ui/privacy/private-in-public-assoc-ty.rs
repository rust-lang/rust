// Private types and traits are not allowed in interfaces of associated types.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults, existential_type)]

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

        type Exist;
        fn infer_exist() -> Self::Exist;
    }
    impl PubTr for u8 {
        type Alias1 = Priv;
        //~^ ERROR private type `m::Priv` in public interface

        existential type Exist: PrivTr;
        //~^ ERROR private trait `m::PrivTr` in public interface
        fn infer_exist() -> Self::Exist { Priv }
    }
}

fn main() {}
