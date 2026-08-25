//@ compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]
use std::future::Future;

// Test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/235

// These are cases where an opaque types become "stranded" due to
// some errors. Make sure we don't ICE in either case.

// Case 1: `impl Send` is stranded
fn foo() -> impl ?Future<Output = impl Send> {
    //~^ ERROR bound modifier `?` can only be applied to `Sized`
    //~| ERROR bound modifier `?` can only be applied to `Sized`
    ()
}

// Case 2: `Assoc = impl Trait` is stranded
trait Trait {}
impl Trait for i32 {}

fn produce() -> impl Trait<Assoc = impl Trait> {
    //~^ ERROR associated type `Assoc` not found for `Trait`
    //~| ERROR associated type `Assoc` not found for `Trait`
    16
}

// Case 3: `impl Trait` is stranded
fn ill_formed_string() -> String<impl Trait> {
    //~^ ERROR struct takes 0 generic arguments but 1 generic argument was supplied
   String::from("a string")
}

// Case 4: TAIT variant of Case 1 to 3
type Foo = impl ?Future<Output = impl Send>;
//~^ ERROR unconstrained opaque type
//~| ERROR unconstrained opaque type
//~| ERROR bound modifier `?` can only be applied to `Sized`
//~| ERROR bound modifier `?` can only be applied to `Sized`

type Produce =  impl Trait<Assoc = impl Trait>;
//~^ ERROR unconstrained opaque type
//~| ERROR unconstrained opaque type
//~| ERROR associated type `Assoc` not found for `Trait`
//~| ERROR associated type `Assoc` not found for `Trait`

type IllFormedString =  String<impl Trait>;
//~^ ERROR unconstrained opaque type
//~| ERROR struct takes 0 generic arguments but 1 generic argument was supplied

fn main() {}
