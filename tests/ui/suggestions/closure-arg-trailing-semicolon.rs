//! Check that a trailing semicolon in a closure body that makes the closure return `()` and fail
//! a trait bound on the function's generic param gets a "remove this semicolon" suggestion, like
//! `-> impl Trait` function bodies already do.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/54771> (closure case).

trait Bar {}
impl Bar for u8 {}
//~^ HELP the trait `Bar` is implemented for `u8`
//~| HELP the trait `Bar` is implemented for `u8`
//~| HELP the trait `Bar` is implemented for `u8`
//~| HELP the trait `Bar` is implemented for `u8`
//~| HELP the trait `Bar` is implemented for `u8`

fn bar<R: Bar>(_: impl Fn() -> R) {}

fn two<R: Bar>(_: impl Fn() -> (), _: impl Fn() -> R) {}

fn unrelated<R: Bar>(_: impl Fn() -> ()) -> R {
    loop {}
}

struct S;
impl S {
    fn run<R: Bar>(&self, _: impl Fn() -> R) {}
}

trait Callable<T: Bar> {
    const CALL: fn();
}

impl<T: Bar> Callable<T> for () {
    const CALL: fn() = || {};
}

fn main() {
    bar(|| { 5u8; });
    //~^ ERROR the trait bound `(): Bar` is not satisfied
    //~| HELP remove this semicolon

    S.run(|| { 5u8; });
    //~^ ERROR the trait bound `(): Bar` is not satisfied
    //~| HELP remove this semicolon

    let c = || { 5u8; };
    //~^ HELP remove this semicolon
    bar(c);
    //~^ ERROR the trait bound `(): Bar` is not satisfied

    // No suggestion: the last statement isn't an expression with a semicolon.
    bar(|| { fn why() {} });
    //~^ ERROR the trait bound `(): Bar` is not satisfied

    // No suggestion: the tail expression's type doesn't implement `Bar`.
    bar(|| { "x"; });
    //~^ ERROR the trait bound `(): Bar` is not satisfied

    // No suggestion: the closure body is empty.
    bar(|| {});
    //~^ ERROR the trait bound `(): Bar` is not satisfied

    // Only the second closure returns the type the failing bound is on.
    two(|| { 5u8; }, || { 7u8; });
    //~^ ERROR the trait bound `(): Bar` is not satisfied
    //~| HELP remove this semicolon

    // No suggestion: `R` is the return type of `unrelated` and is inferred as `()` from the
    // expected type, so keeping the closure's value would not satisfy the bound.
    let _: () = unrelated(|| { 5u8; });
    //~^ ERROR the trait bound `(): Bar` is not satisfied

    // No suggestion: the failing clause belongs to an associated const, which has no signature to
    // match a closure argument against.
    <() as Callable<()>>::CALL();
    //~^ ERROR the trait bound `(): Bar` is not satisfied
}
