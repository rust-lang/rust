// This test ensures that the `for-loops-over-fallibles` lint doesn't suggest
// removing `next`.
// ref. <https://github.com/rust-lang/rust-clippy/issues/16133>

#![forbid(for_loops_over_fallibles)]
//~^ NOTE: the lint level is defined here

fn main() {
    macro_rules! mac {
        (next $e:expr) => {
            $e.iter().next()
        };
    }

    for _ in mac!(next [1, 2]) {}
    //~^ ERROR: for loop over an `Option`. This is more readably written as an `if let` statement
    //~| NOTE: in this expansion of desugaring of `for` loop
    //~| NOTE: in this expansion of desugaring of `for` loop
    //~| HELP: to check pattern in a loop use `while let`
    //~| HELP: consider using `if let` to clear intent
}
