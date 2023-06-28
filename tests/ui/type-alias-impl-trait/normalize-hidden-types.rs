// Regression test for #112691
//
// revisions: current next
// [next] compile-flags: -Ztrait-solver=next
// [next] check-pass
// [current]: known-bug: #112691

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
    fn define() -> Opaque {
        //[current]~^ ERROR concrete type differs
        dyn_hoops::<_>(0)
    }
}

mod typeof_2 {
    use super::*;
    type Opaque = impl Sized;
    fn define_1() -> Opaque { dyn_hoops::<_>(0) }
    //[current]~^ ERROR concrete type differs
    fn define_2() -> Opaque { dyn_hoops::<u8>(0) }
}

mod typeck {
    use super::*;
    type Opaque = impl Sized;
    fn define() -> Option<Opaque> {
        let _: Opaque = dyn_hoops::<_>(0);
        let _: Opaque = dyn_hoops::<u8>(0);
        //[current]~^ ERROR mismatched types
        None
    }
}

mod borrowck {
    use super::*;
    type Opaque = impl Sized;
    fn define() -> Option<Opaque> {
        let _: Opaque = dyn_hoops::<_>(0);
        //[current]~^ ERROR concrete type differs
        None
    }
}

fn main() {}
