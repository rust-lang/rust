//@ check-pass

// Test when deferring repeat expr checks to end of typechecking whether they're
// checked before integer fallback occurs. We accomplish this by having the repeat
// expr check allow inference progress on an ambiguous goal, where the ambiguous goal
// would fail if the inference variable was fallen back to `i32`. This test will
// pass if we check repeat exprs before integer fallback.

use std::marker::PhantomData;
struct Foo<T>(PhantomData<T>);

impl Clone for Foo<u32> {
    fn clone(&self) -> Self {
        Foo(PhantomData)
    }
}
impl Copy for Foo<u32> {}

trait Trait {}

// Two impls just to ensure that `?int: Trait` wont itself succeed by unifying with
// a self type on an impl here. It also ensures that integer fallback would actually
// be valid for all of the stalled goals incase that's ever something we take into account.
impl Trait for i32 {}
impl Trait for u32 {}

fn make_goal<T: Trait>(_: &T) {}
fn tie<T>(_: &T, _: &[Foo<T>; 2]) {}

fn main() {
    let a = 1;
    // `?int: Trait`
    make_goal(&a);

    // Deferred `Foo<?int>: Copy` requirement
    let b: [Foo<_>; 2] = [Foo(PhantomData); _];
    tie(&a, &b);

    // If fallback doesn't occur:
    // - `Foo<?int>; 2`is > 1, needs copy
    // - `Foo<?int>: Copy` infers `?int=u32`
    // - stalled goal `?int: Trait` can now make progress and succeed

    // If fallback occurs:
    // - `Foo<i32>; 2` is > 1, needs copy
    // - `Foo<i32>: Copy` doesn't hold -> error
}
