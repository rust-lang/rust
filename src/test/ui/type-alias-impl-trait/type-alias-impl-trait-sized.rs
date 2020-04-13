// check-pass

#![feature(type_alias_impl_trait)]

type A = impl Sized;
fn f1() -> A { 0 }

type B = impl ?Sized;
fn f2() -> &'static B { &[0] }

type C = impl ?Sized + 'static;
fn f3() -> &'static C { &[0] }

type D = impl ?Sized;
fn f4() -> &'static D { &1 }

fn main() {}
