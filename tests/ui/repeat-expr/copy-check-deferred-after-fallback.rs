#![feature(generic_arg_infer)]

// Test that would start passing if we defer repeat expr copy checks to end of
// typechecking and they're checked after integer fallback occurs. We accomplish
// this by contriving a situation where integer fallback allows progress to be
// made on a trait goal that infers the length of a repeat expr.

use std::marker::PhantomData;

struct NotCopy;

trait Trait<const N: usize> {}

impl Trait<2> for u32 {}
impl Trait<1> for i32 {}

fn make_goal<T: Trait<N>, const N: usize>(_: &T, _: [NotCopy; N]) {}

fn main() {
    let a = 1;
    let b = [NotCopy; _];
    //~^ ERROR: type annotations needed

    // a is of type `?y`
    // b is of type `[NotCopy; ?x]`
    // there is a goal ?y: Trait<?x>` with two candidates:
    // - `i32: Trait<1>`, ?y=i32 ?x=1 which doesnt require `NotCopy: Copy`
    // - `u32: Trait<2>` ?y=u32 ?x=2 which requires `NotCopy: Copy`
    make_goal(&a, b);

    // final repeat expr checks:
    //
    // `NotCopy; ?x`
    // - succeeds if fallback happens before repeat exprs as `i32: Trait<?x>` infers `?x=1`
    // - fails if repeat expr checks happen first as `?x` is unconstrained so cannot be
    //    structurally resolved
}
