// Private types and traits are not allowed in interfaces of associated types.
// This test also ensures that the checks are performed even inside private modules.

#![feature(associated_type_defaults)]
#![feature(impl_trait_in_assoc_type)]

mod m {
    struct Priv;
    trait PrivTr {}
    impl PrivTr for Priv {}
    pub trait PubTrAux1<T> {}
    pub trait PubTrAux2 {
        type A;
    }
    impl<T> PubTrAux1<T> for u8 {}
    impl PubTrAux2 for u8 {
        type A = Priv;
        //~^ ERROR private type `Priv` in public interface
    }

    // "Private-in-public in associated types is hard error" in RFC 2145
    // applies only to the aliased types, not bounds.
    pub trait PubTr {
        type Alias1: PrivTr;
        //~^ WARN trait `PrivTr` is more private than the item `PubTr::Alias1`
        type Alias2: PubTrAux1<Priv> = u8;
        //~^ WARN type `Priv` is more private than the item `PubTr::Alias2`
        type Alias3: PubTrAux2<A = Priv> = u8;
        //~^ WARN type `Priv` is more private than the item `PubTr::Alias3`

        type Alias4 = Priv;
        //~^ ERROR private type `Priv` in public interface

        type Exist;
        fn infer_exist() -> Self::Exist;
    }
    impl PubTr for u8 {
        type Alias1 = Priv;
        //~^ ERROR private type `Priv` in public interface

        type Exist = impl PrivTr;
        //~^ ERROR private trait `PrivTr` in public interface
        fn infer_exist() -> Self::Exist {
            Priv
        }
    }
}

fn main() {}
