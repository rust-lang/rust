// Test diagnostic for the case where a trait is not implemented for `!`. If it is implemented
// for `()`, we want to add a note saying that this might be caused by a breaking change in the
// compiler.
//
//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@[e2021] run-pass
#![expect(dependency_on_unit_never_type_fallback, unused)]

trait OnlyUnit {}

impl OnlyUnit for () {}
//[e2024]~^ help: trait `OnlyUnit` is implemented for `()`

fn requires_unit(_: impl OnlyUnit) {}
//[e2024]~^ note: required by this bound in `requires_unit`
//[e2024]~| note: required by a bound in `requires_unit`


trait OnlyU32 {}

impl OnlyU32 for u32 {}
//[e2024]~^ help: the trait `OnlyU32` is implemented for `u32`

fn requires_u32(_: impl OnlyU32) {}
//[e2024]~^ note: required by this bound in `requires_u32`
//[e2024]~| note: required by a bound in `requires_u32`


trait Nothing {}
//[e2024]~^ help: this trait has no implementations, consider adding one

fn requires_nothing(_: impl Nothing) {}
//[e2024]~^ note: required by this bound in `requires_nothing`
//[e2024]~| note: required by a bound in `requires_nothing`

fn main() {
    let x = return;
    requires_unit(x);
    //[e2024]~^ error: the trait bound `!: OnlyUnit` is not satisfied
    //[e2024]~| note: the trait `OnlyUnit` is not implemented for `!`
    //[e2024]~| note: this error might have been caused by changes to Rust's type-inference algorithm (see issue #148922 <https://github.com/rust-lang/rust/issues/148922> for more information)
    //[e2024]~| note: required by a bound introduced by this call
    //[e2024]~| help: you might have intended to use the type `()`

    #[cfg(e2024)]
    requires_u32(x);
    //[e2024]~^ error: the trait bound `!: OnlyU32` is not satisfied
    //[e2024]~| note: the trait `OnlyU32` is not implemented for `!`
    //[e2024]~| note: required by a bound introduced by this call

    #[cfg(e2024)]
    requires_nothing(x);
    //[e2024]~^ error: the trait bound `!: Nothing` is not satisfied
    //[e2024]~| note: the trait `Nothing` is not implemented for `!`
    //[e2024]~| note: required by a bound introduced by this call
}
