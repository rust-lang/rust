//@ check-pass
//@ edition:2021

#![warn(non_local_definitions)]

trait Global {}

fn main() {
    trait Local {};

    impl<T: Local> Global for Vec<T> { }
    //~^ WARN non-local `impl` definition
}

trait Uto7 {}
trait Uto8 {}

struct Test;

fn bad() {
    struct Local;
    impl Uto7 for Test where Local: std::any::Any {}
    //~^ WARN non-local `impl` definition

    impl<T> Uto8 for T {}
    //~^ WARN non-local `impl` definition
}

struct UwU<T>(T);

fn fun() {
    #[derive(Debug)]
    struct OwO;
    impl Default for UwU<OwO> {
    //~^ WARN non-local `impl` definition
        fn default() -> Self {
            UwU(OwO)
        }
    }
}

fn meow() {
    #[derive(Debug)]
    struct Cat;
    impl AsRef<Cat> for () {
    //~^ WARN non-local `impl` definition
        fn as_ref(&self) -> &Cat { &Cat }
    }
}

struct G;

fn fun2() {
    #[derive(Debug, Default)]
    struct B;
    impl PartialEq<B> for G {
    //~^ WARN non-local `impl` definition
        fn eq(&self, _: &B) -> bool {
            true
        }
    }
}

struct Wrap<T>(T);

impl Wrap<Wrap<Wrap<()>>> {}

fn rawr() {
    struct Lion;

    impl From<Wrap<Wrap<Lion>>> for () {
    //~^ WARN non-local `impl` definition
        fn from(_: Wrap<Wrap<Lion>>) -> Self {
            todo!()
        }
    }

    impl From<()> for Wrap<Lion> {
    //~^ WARN non-local `impl` definition
        fn from(_: ()) -> Self {
            todo!()
        }
    }
}

fn side_effects() {
    dbg!(().as_ref()); // prints `Cat`
    dbg!(UwU::default().0);
    let _ = G::eq(&G, dbg!(&<_>::default()));
}
