#![feature(type_alias_impl_trait)]

use std::iter;

fn frpit1() -> impl Iterator<Item: Copy, Item: Send> {
    iter::empty()
    //~^ ERROR type annotations needed
}
fn frpit2() -> impl Iterator<Item: Copy, Item: Copy> {
    iter::empty()
    //~^ ERROR type annotations needed
}
fn frpit3() -> impl Iterator<Item: 'static, Item: 'static> {
    iter::empty()
    //~^ ERROR type annotations needed
}

type ETAI1<T: Iterator<Item: Copy, Item: Send>> = impl Copy;
//~^ ERROR unconstrained opaque type
type ETAI2<T: Iterator<Item: Copy, Item: Copy>> = impl Copy;
//~^ ERROR unconstrained opaque type
type ETAI3<T: Iterator<Item: 'static, Item: 'static>> = impl Copy;
//~^ ERROR unconstrained opaque type

type ETAI4 = impl Iterator<Item: Copy, Item: Send>;
//~^ ERROR unconstrained opaque type
type ETAI5 = impl Iterator<Item: Copy, Item: Copy>;
//~^ ERROR unconstrained opaque type
type ETAI6 = impl Iterator<Item: 'static, Item: 'static>;
//~^ ERROR unconstrained opaque type

fn mismatch() -> impl Iterator<Item: Copy, Item: Send> { //~ ERROR [E0277]
    iter::empty::<*const ()>()
}

fn mismatch_2() -> impl Iterator<Item: Copy, Item: Send> { //~ ERROR [E0277]
    iter::empty::<String>()
}

fn main() {}
