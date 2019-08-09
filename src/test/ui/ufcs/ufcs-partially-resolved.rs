#![feature(associated_type_defaults)]

trait Tr {
    type Y = u16;
    fn Y() {}
}
impl Tr for u8 {}

trait Dr {
    type X = u16;
    fn Z() {}
}
impl Dr for u8 {}

enum E { Y }
type A = u32;

fn main() {
    let _: <u8 as Tr>::N; //~ ERROR cannot find associated type `N` in trait `Tr`
    let _: <u8 as E>::N; //~ ERROR cannot find associated type `N` in enum `E`
    let _: <u8 as A>::N; //~ ERROR cannot find associated type `N` in `A`
    <u8 as Tr>::N; //~ ERROR cannot find method or associated constant `N` in trait `Tr`
    <u8 as E>::N; //~ ERROR cannot find method or associated constant `N` in enum `E`
    <u8 as A>::N; //~ ERROR cannot find method or associated constant `N` in `A`
    let _: <u8 as Tr>::Y; // OK
    let _: <u8 as E>::Y; //~ ERROR expected associated type, found variant `E::Y`
    <u8 as Tr>::Y; // OK
    <u8 as E>::Y; //~ ERROR expected method or associated constant, found unit variant `E::Y`

    let _: <u8 as Tr>::N::NN; //~ ERROR cannot find associated type `N` in trait `Tr`
    let _: <u8 as E>::N::NN; //~ ERROR cannot find associated type `N` in enum `E`
    let _: <u8 as A>::N::NN; //~ ERROR cannot find associated type `N` in `A`
    <u8 as Tr>::N::NN; //~ ERROR cannot find associated type `N` in trait `Tr`
    <u8 as E>::N::NN; //~ ERROR cannot find associated type `N` in enum `E`
    <u8 as A>::N::NN; //~ ERROR cannot find associated type `N` in `A`
    let _: <u8 as Tr>::Y::NN; //~ ERROR ambiguous associated type
    let _: <u8 as E>::Y::NN; //~ ERROR expected associated type, found variant `E::Y`
    <u8 as Tr>::Y::NN; //~ ERROR no associated item named `NN` found for type `<u8 as Tr>::Y`
    <u8 as E>::Y::NN; //~ ERROR expected associated type, found variant `E::Y`

    let _: <u8 as Tr::N>::NN; //~ ERROR cannot find associated type `NN` in `Tr::N`
    let _: <u8 as E::N>::NN; //~ ERROR cannot find associated type `NN` in `E::N`
    let _: <u8 as A::N>::NN; //~ ERROR cannot find associated type `NN` in `A::N`
    <u8 as Tr::N>::NN; //~ ERROR cannot find method or associated constant `NN` in `Tr::N`
    <u8 as E::N>::NN; //~ ERROR cannot find method or associated constant `NN` in `E::N`
    <u8 as A::N>::NN; //~ ERROR cannot find method or associated constant `NN` in `A::N`
    let _: <u8 as Tr::Y>::NN; //~ ERROR cannot find associated type `NN` in `Tr::Y`
    let _: <u8 as E::Y>::NN; //~ ERROR failed to resolve: `Y` is a variant, not a module
    <u8 as Tr::Y>::NN; //~ ERROR cannot find method or associated constant `NN` in `Tr::Y`
    <u8 as E::Y>::NN; //~ ERROR failed to resolve: `Y` is a variant, not a module

    let _: <u8 as Dr>::Z; //~ ERROR expected associated type, found method `Dr::Z`
    <u8 as Dr>::X; //~ ERROR expected method or associated constant, found associated type `Dr::X`
    let _: <u8 as Dr>::Z::N; //~ ERROR expected associated type, found method `Dr::Z`
    <u8 as Dr>::X::N; //~ ERROR no associated item named `N` found for type `<u8 as Dr>::X`
}
