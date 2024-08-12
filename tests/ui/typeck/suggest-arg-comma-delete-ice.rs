//! Previously, we tried to remove extra arg commas when providing extra arg removal suggestions.
//! One of the edge cases is having to account for an arg that has a closing delimiter `)`
//! following it. However, the previous suggestion code assumed that the delimiter is in fact
//! exactly the 1-byte `)` character. This assumption was proven incorrect, because we recover
//! from Unicode-confusable delimiters in the parser, which means that the ending delimiter could be
//! a multi-byte codepoint that looks *like* a `)`. Subtracing 1 byte could land us in the middle of
//! a codepoint, triggering a codepoint boundary assertion.
//!
//! issue: rust-lang/rust#128717

fn main() {
    // The following example has been modified from #128717 to remove irrelevant Unicode as they do
    // not otherwise partake in the right delimiter calculation causing the codepoint boundary
    // assertion.
    main(rahh）;
    //~^ ERROR unknown start of token
    //~| ERROR this function takes 0 arguments but 1 argument was supplied
    //~| ERROR cannot find value `rahh` in this scope
}
