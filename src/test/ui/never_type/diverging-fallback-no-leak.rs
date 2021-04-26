fn make_unit() {}

trait Test {}
impl Test for i32 {}
impl Test for () {}

fn unconstrained_arg<T: Test>(_: T) {}

fn main() {
    // Here the type variable falls back to `!`,
    // and hence we get a type error:
    unconstrained_arg(return); //~ ERROR trait bound `!: Test` is not satisfied
}
