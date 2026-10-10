#![allow(dead_code)]
#![deny(unused_results, unused_must_use)]
//~^ NOTE: the lint level is defined here
//~| NOTE: the lint level is defined here

use std::ops::ControlFlow;

#[must_use]
enum MustUse { Test }

#[must_use = "some message"]
enum MustUseMsg { Test2 }

enum Nothing {}

fn foo<T>() -> T { panic!() }

fn bar() -> isize { return foo::<isize>(); }
fn baz() -> MustUse { return foo::<MustUse>(); }
fn qux() -> MustUseMsg { return foo::<MustUseMsg>(); }

#[allow(unused_results)]
fn test() {
    foo::<isize>();
    foo::<MustUse>(); //~ ERROR: unused `MustUse` that must be used
    foo::<MustUseMsg>(); //~ ERROR: unused `MustUseMsg` that must be used
    //~^ NOTE: some message
}

#[allow(unused_results, unused_must_use)]
fn test2() {
    foo::<isize>();
    foo::<MustUse>();
    foo::<MustUseMsg>();
}

fn main() {
    foo::<isize>(); //~ ERROR: unused result of type `isize`
    foo::<MustUse>(); //~ ERROR: unused `MustUse` that must be used
    foo::<MustUseMsg>(); //~ ERROR: unused `MustUseMsg` that must be used
    //~^ NOTE: some message

    let _ = foo::<isize>();
    let _ = foo::<MustUse>();
    let _ = foo::<MustUseMsg>();

    // "trivial" types
    ();
    ((), ());
    Ok::<(), Nothing>(());
    ControlFlow::<Nothing>::Continue(());
    ((), Ok::<(), Nothing>(()), ((((), ())), ((),)));
    foo::<Nothing>();

    ((), 1); //~ ERROR: unused result of type `((), i32)`
    (1, ()); //~ ERROR: unused result of type `(i32, ())`
}
