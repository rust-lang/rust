//@ check-pass

#![feature(type_alias_impl_trait)]

type A = impl Sized;
#[define_opaques(A)]
fn f1() -> A {
    0
}

type B = impl ?Sized;
#[define_opaques(B)]
fn f2() -> &'static B {
    &[0]
}

type C = impl ?Sized + 'static;
#[define_opaques(C)]
fn f3() -> &'static C {
    &[0]
}

type D = impl ?Sized;
#[define_opaques(D)]
fn f4() -> &'static D {
    &1
}

fn main() {}
