//@ run-rustfix
#![allow(dead_code)]

struct S;

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
fn bar<A, B, C, D>(_: Many<A, B, C, D>) {}

fn take_two(_: bool, _: bool) {}
fn take_three(_: bool, _: bool, _: Many<i32, Many<(), i32, S, S>, i32, i32>) {}

fn main() {
    let _ = bar(Many<i32, Many<(), i32, S, S>, i32, i32>::new());
    //~^ ERROR expected expression

    // These are unambiguously comparisons and must keep compiling.
    let (a, b, c, d) = (1, 2, 3, 4);
    take_two(a < b, c > (d));
    take_two(a < b, c > ::std::primitive::i32::MAX);

    // An argument preceded by genuine comparisons.
    take_three(a < b, c > (d), Many<i32, Many<(), i32, S, S>, i32, i32>::new());
    //~^ ERROR expected expression
}
