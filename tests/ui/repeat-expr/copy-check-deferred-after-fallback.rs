#![feature(generic_arg_infer)]

// Test when deferring repeat expr copy checks to end of typechecking whether they're
// checked before integer fallback occurs or not. We accomplish this by having a repeat
// count that can only be inferred after integer fallback has occured. This test will
// pass if we were to check repeat exprs after integer fallback.

use std::marker::PhantomData;
struct Foo<T>(PhantomData<T>);

// We impl Copy/Clone for multiple (but not all) substitutions
// to ensure that `Foo<?int>: Copy` can't be proven on the basis
// of there only being one applying impl.
impl Clone for Foo<u32> {
    fn clone(&self) -> Self {
        Foo(PhantomData)
    }
}
impl Clone for Foo<i32> {
    fn clone(&self) -> Self {
        Foo(PhantomData)
    }
}
impl Copy for Foo<u32> {}
impl Copy for Foo<i32> {}

trait Trait<const N: usize> {}

// We impl `Trait` for both `i32` and `u32` to avoid being able
// to prove `?int: Trait<?n>` from there only being one impl.
impl Trait<1> for i32 {}
impl Trait<2> for u32 {}

fn tie_and_make_goal<const N: usize, T: Trait<N>>(_: &T, _: &[Foo<T>; N]) {}

fn main() {
    let a = 1;
    // Deferred repeat expr `Foo<?int>; ?n`
    let b = [Foo(PhantomData); _];
    //~^ ERROR: type annotations needed for `[Foo<{integer}>; _]`

    // Introduces a `?int: Trait<?n>` goal
    tie_and_make_goal(&a, &b);

    // If fallback doesn't occur:
    // - `Foo<?int>; ?n`is ambig as repeat count is unknown -> error

    // If fallback occurs:
    // - `?int` inferred to `i32`
    // - `?int: Trait<?n>` becomes `i32: Trait<?n>` wihhc infers `?n=1`
    // - Repeat expr check `Foo<?int>; ?n` is now `Foo<i32>; 1`
    // - `Foo<i32>; 1` doesn't require `Foo<i32>: Copy`
}
