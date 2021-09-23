// revisions: nofallback fallback
//[nofallback] check-pass

#![cfg_attr(fallback, feature(never_type, never_type_fallback))]

fn make_unit() {}

trait Test {}
impl Test for i32 {}
impl Test for () {}

fn unconstrained_arg<T: Test>(_: T) {}

fn main() {
    // Here the type variable falls back to `!`,
    // and hence we get a type error.
    unconstrained_arg(return);
    //[fallback]~^ ERROR trait bound `!: Test` is not satisfied
}
