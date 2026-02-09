//@ edition: 2024
//@ compile-flags: -Znext-solver --diagnostic-width=300

// Previously we check stalled coroutine obligations after borrowck pass.
// And we wrongly assume that these obligations hold in borrowck which leads to
// silent normalization failures.
// In the next solver, we register opaques types via `NormalizesTo` goals.
// So these failures also cause those opaques types not registered in storage.
//
// Regression test for #151322 and #151323.

#![feature(type_alias_impl_trait)]
#![feature(negative_impls)]
#![feature(auto_traits)]

fn stalled_copy_clone() {
    type T = impl Copy;
    let foo: T = async {};
    //~^ ERROR: the trait bound

    type U = impl Clone;
    let bar: U = async {};
    //~^ ERROR: the trait bound
}

auto trait Valid {}
struct False;
impl !Valid for False {}

fn stalled_auto_traits() {
    type T = impl Valid;
    let a = False;
    let foo: T = async { a };
    //~^ ERROR: the trait bound `False: Valid` is not satisfied
}


trait Trait {
    fn stalled_send(&self, b: *mut ()) -> impl Future + Send {
    //~^ ERROR: type mismatch resolving
    //~| ERROR: type mismatch resolving
        async move {
            //~^ ERROR: type mismatch resolving
            b
        }
    }
}


fn main() {}
