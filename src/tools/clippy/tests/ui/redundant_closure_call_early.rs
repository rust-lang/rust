// non rustfixable, see redundant_closure_call_fixable.rs

#![warn(clippy::redundant_closure_call)]

fn main() {
    let mut i = 1;

    // lint here
    let mut k = (|m| m + 1)(i);
    //~^ ERROR: try not to call a closure in the expression where it is declared
    //~| NOTE: `-D clippy::redundant-closure-call` implied by `-D warnings`

    // lint here
    k = (|a, b| a * b)(1, 5);
    //~^ ERROR: try not to call a closure in the expression where it is declared

    // don't lint these
    #[allow(clippy::needless_return)]
    (|| return 2)();
    (|| -> Option<i32> { None? })();
    #[allow(clippy::try_err)]
    (|| -> Result<i32, i32> { Err(2)? })();
}
