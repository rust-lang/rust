//@ revisions: nofallback fallback
//@[nofallback] check-pass

#![cfg_attr(fallback, feature(never_type, never_type_fallback))]

fn make_unit() {}

trait Test {}
impl Test for i32 {}
impl Test for () {}

fn unconstrained_arg<T: Test>(_: T) {}

fn main() {
    //[nofallback]~^ warn: this function depends on never type fallback being `()`
    //[nofallback]~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!

    // Here the type variable falls back to `!`,
    // and hence we get a type error.
    unconstrained_arg(return);
    //[fallback]~^ ERROR trait bound `!: Test` is not satisfied
}
