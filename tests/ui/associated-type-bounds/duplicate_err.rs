#![feature(associated_const_equality, type_alias_impl_trait, return_type_notation)]
#![allow(refining_impl_trait_internal)]

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

trait Trait {
    type Gat<T>;

    const ASSOC: i32;

    fn foo() -> impl Sized;
}

impl Trait for () {
    type Gat<T> = ();

    const ASSOC: i32 = 3;

    fn foo() {}
}

impl Trait for u32 {
    type Gat<T> = ();

    const ASSOC: i32 = 4;

    fn foo() -> u32 {
        42
    }
}

fn uncallable(_: impl Iterator<Item = i32, Item = u32>) {}

fn uncallable_const(_: impl Trait<ASSOC = 3, ASSOC = 4>) {}

fn uncallable_rtn(_: impl Trait<foo(..): Trait<ASSOC = 3>, foo(..): Trait<ASSOC = 4>>) {}

type MustFail = dyn Iterator<Item = i32, Item = u32>;
//~^ ERROR conflicting associated type bounds

trait Trait2 {
    const ASSOC: u32;
}

type MustFail2 = dyn Trait2<ASSOC = 3u32, ASSOC = 4u32>;
//~^ ERROR conflicting associated type bounds

fn main() {
    uncallable(iter::empty::<u32>()); //~ ERROR [E0271]
    uncallable(iter::empty::<i32>()); //~ ERROR [E0271]
    uncallable_const(()); //~ ERROR [E0271]
    uncallable_const(4u32); //~ ERROR [E0271]
    uncallable_rtn(()); //~ ERROR [E0271]
    uncallable_rtn(17u32); //~ ERROR [E0271]
}
