//@ check-pass

#![feature(generic_arg_infer)]

// Test that if we defer repeat expr copy checks to end of typechecking they're
// checked before integer fallback occurs. We accomplish this by contriving a
// situation where we have a goal that can be proven either via another repeat expr
// check or by integer fallback. In the integer fallback case an array length would
// be inferred to `2` requiring `NotCopy: Copy`, and in the repeat expr case it would
// be inferred to `1`.

use std::marker::PhantomData;

struct NotCopy;

struct Foo<T>(PhantomData<T>);

impl Clone for Foo<u32> {
    fn clone(&self) -> Self {
        Foo(PhantomData)
    }
}

impl Copy for Foo<u32> {}

fn tie<T>(_: &T, _: [Foo<T>; 2]) {}

trait Trait<const N: usize> {}

impl Trait<2> for i32 {}
impl Trait<1> for u32 {}

fn make_goal<T: Trait<N>, const N: usize>(_: &T, _: [NotCopy; N]) {}

fn main() {
    let a = 1;
    let b: [Foo<_>; 2] = [Foo(PhantomData); _];
    tie(&a, b);
    let c = [NotCopy; _];

    // a is of type `?y`
    // b is of type `[Foo<?y>; 2]`
    // c is of type `[NotCopy; ?x]`
    // there is a goal ?y: Trait<?x>` with two candidates:
    // - `i32: Trait<2>`, ?y=i32 ?x=2 which requires `NotCopy: Copy` when expr checks happen
    // - `u32: Trait<1>` ?y=u32 ?x=1 which doesnt require `NotCopy: Copy`
    make_goal(&a, c);

    // final repeat expr checks:
    //
    // `Foo<?y>; 2`
    // - Foo<?y>: Copy
    // - requires ?y=u32
    //
    // `NotCopy; ?x`
    // - fails if fallback happens before repeat exprs as `i32: Trait<?x>` infers `?x=2`
    // - succeeds if repeat expr checks happen first as `?y=u32` means `u32: Trait<?x>`
    //    infers `?x=1`
}
