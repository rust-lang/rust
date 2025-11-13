//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//
//@[e2021] check-pass


fn make_unit() {}

trait Test {}
impl Test for i32 {}
impl Test for () {}

fn unconstrained_arg<T: Test>(_: T) {}

#[cfg_attr(e2021, expect(dependency_on_unit_never_type_fallback))]
fn main() {
    // Here the type variable falls back to `!`,
    // and hence we get a type error.
    unconstrained_arg(return);
    //[e2024]~^ error: trait bound `!: Test` is not satisfied
}
