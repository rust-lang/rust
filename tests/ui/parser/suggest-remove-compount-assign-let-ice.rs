//! Previously we would try to issue a suggestion for `let x <op>= 1`, i.e. a compound assignment
//! within a `let` binding, to remove the `<op>`. The suggestion code unfortunately incorrectly
//! assumed that the `<op>` is an exactly-1-byte ASCII character, but this assumption is incorrect
//! because we also recover Unicode-confusables like `➖=` as `-=`. In this example, the suggestion
//! code used a `+ BytePos(1)` to calculate the span of the `<op>` codepoint that looks like `-` but
//! the mult-byte Unicode look-alike would cause the suggested removal span to be inside a
//! multi-byte codepoint boundary, triggering a codepoint boundary assertion.
//!
//! issue: rust-lang/rust#128845

fn main() {
    // Adapted from #128845 but with irrelevant components removed and simplified.
    let x ➖= 1;
    //~^ ERROR unknown start of token: \u{2796}
    //~| ERROR: can't reassign to an uninitialized variable
}
