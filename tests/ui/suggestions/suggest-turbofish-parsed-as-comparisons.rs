struct Many<A, B, C, D> {
    a: A,
    b: B,
    c: C,
    d: D,
}
impl<A, B, C, D> Many<A, B, C, D> {
    fn new() -> Self {
        todo!()
    }
}
fn bar<T>(_: Many<T, T, T, T>) {}

fn take_two(_: bool, _: bool) {}

fn main() {
    let _ = bar(Many<i32, Many<(), i32, S, S>, i32, i32>::new());
    //~^ ERROR generic args in this position require the turbofish syntax

    // These are unambiguously comparisons and must keep compiling.
    let (a, b, c, d) = (1, 2, 3, 4);
    take_two(a < b, c > (d));
    take_two(a < b, c > ::std::primitive::i32::MAX);
}
