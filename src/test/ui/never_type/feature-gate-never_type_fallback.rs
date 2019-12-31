// This is a feature gate test for `never_type_fallback`.
// It works by using a scenario where the type fall backs to `()` rather than ´!`
// in the case where `#![feature(never_type_fallback)]` would change it to `!`.

fn main() {}

trait T {}

fn should_ret_unit() -> impl T {
    //~^ ERROR the trait bound `(): T` is not satisfied
    panic!()
}
