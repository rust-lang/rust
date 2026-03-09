// Test diagnostic for the case where a trait is not implemented for `!`. If it is implemented
// for `()`, we want to add a note saying that this might be caused by a breaking change in the
// compiler.
#![expect(dependency_on_unit_never_type_fallback, unused)]

trait OnlyUnit {}

impl OnlyUnit for () {}
//~^ help: trait `OnlyUnit` is implemented for `()`

fn requires_unit(_: impl OnlyUnit) {}
//~^ note: required by this bound in `requires_unit`
//~| note: required by a bound in `requires_unit`

trait OnlyU32 {}

impl OnlyU32 for u32 {}
//~^ help: the trait `OnlyU32` is implemented for `u32`

fn requires_u32(_: impl OnlyU32) {}
//~^ note: required by this bound in `requires_u32`
//~| note: required by a bound in `requires_u32`

trait Nothing {}
//~^ help: this trait has no implementations, consider adding one

fn requires_nothing(_: impl Nothing) {}
//~^ note: required by this bound in `requires_nothing`
//~| note: required by a bound in `requires_nothing`

fn main() {
    let x = return;
    requires_unit(x);
    //~^ error: the trait bound `!: OnlyUnit` is not satisfied
    //~| note: the trait `OnlyUnit` is not implemented for `!`
    //~| note: this error might have been caused by changes to Rust's type-inference algorithm (see issue #148922 <https://github.com/rust-lang/rust/issues/148922> for more information)
    //~| note: required by a bound introduced by this call
    //~| help: you might have intended to use the type `()`

    requires_u32(x);
    //~^ error: the trait bound `!: OnlyU32` is not satisfied
    //~| note: the trait `OnlyU32` is not implemented for `!`
    //~| note: required by a bound introduced by this call

    requires_nothing(x);
    //~^ error: the trait bound `!: Nothing` is not satisfied
    //~| note: the trait `Nothing` is not implemented for `!`
    //~| note: required by a bound introduced by this call
}
