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

fn bar<R: Bar>(_: impl Fn() -> R) {}

struct S;
impl S {
    fn run<R: Bar>(&self, _: impl Fn() -> R) {}
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
}
