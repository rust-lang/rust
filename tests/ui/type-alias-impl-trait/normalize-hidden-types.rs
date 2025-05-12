// Regression test for #112691
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ [next] check-pass
//@ [current] known-bug: #112691

#![feature(type_alias_impl_trait)]

trait Trait {
    type Gat<'lt>;
}

impl Trait for u8 {
    type Gat<'lt> = ();
}

fn dyn_hoops<T: Trait>(_: T) -> *const dyn FnOnce(T::Gat<'_>) {
    loop {}
}

mod typeof_1 {
    use super::*;
    type Opaque = impl Sized;
    #[define_opaque(Opaque)]
    fn define() -> Opaque {
        dyn_hoops::<_>(0)
    }
}

mod typeof_2 {
    use super::*;
    type Opaque = impl Sized;
    #[define_opaque(Opaque)]
    fn define_1() -> Opaque { dyn_hoops::<_>(0) }
    #[define_opaque(Opaque)]
    fn define_2() -> Opaque { dyn_hoops::<u8>(0) }
}

mod typeck {
    use super::*;
    type Opaque = impl Sized;
    #[define_opaque(Opaque)]
    fn define() -> Option<Opaque> {
        let _: Opaque = dyn_hoops::<_>(0);
        let _: Opaque = dyn_hoops::<u8>(0);
        None
    }
}

mod borrowck {
    use super::*;
    type Opaque = impl Sized;
    #[define_opaque(Opaque)]
    fn define() -> Option<Opaque> {
        let _: Opaque = dyn_hoops::<_>(0);
        None
    }
}

fn main() {}
