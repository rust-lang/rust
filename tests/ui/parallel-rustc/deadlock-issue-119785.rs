//@ parallel-front-end
//@ compile-flags: -Z threads=200

#![allow(incomplete_features)]
#![feature(
    const_trait_impl,
    effects,
)]

use std::marker::Destruct;

const fn cmp(a: &impl ~const PartialEq) -> bool {
    //~^ ERROR
    a == a
}

const fn wrap(x: impl ~const PartialEq + ~const Destruct)
//~^ ERROR
//~| ERROR
              -> impl ~const PartialEq + ~const Destruct
//~^ ERROR
//~| ERROR
//~| ERROR
//~| ERROR
{
    x
}

#[const_trait]
trait Foo {
    fn foo(&mut self, x: <Self as Index>::Output) -> <Self as Index>::Output;
    //~^ ERROR
    //~| ERROR
}

impl const Foo for () {
    //~^ ERROR
    fn huh() -> impl ~const PartialEq + ~const Destruct + Copy {
        //~^ ERROR
        //~| ERROR
        //~| ERROR
        //~| ERROR
        //~| ERROR
        123
    }
}

const _: () = {
    assert!(cmp(&0xDEADBEEFu32));
    assert!(cmp(&()));
    assert!(wrap(123) == wrap(123));
    assert!(wrap(123) != wrap(456));
    let x = <() as Foo>::huh();
    //~^ ERROR
    assert!(x == x);
};

#[const_trait]
trait T {}
struct S;
impl const T for S {}

const fn rpit() -> impl ~const T { S }

const fn apit(_: impl ~const T + ~const Destruct) {}
//~^ ERROR
//~| ERROR

const fn rpit_assoc_bound() -> impl IntoIterator<Item: ~const T> { Some(S) }

const fn apit_assoc_bound(_: impl IntoIterator<Item: ~const T> + ~From Destruct) {}
//~^ ERROR

fn main() {}
