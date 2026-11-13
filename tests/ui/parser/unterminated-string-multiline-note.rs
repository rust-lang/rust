//@ edition: 2021
//! Forgetting a closing quote makes the lexer run on to the next quote in the file, so the
//! "unterminated double quote string" error is reported far below the actual mistake. Check that
//! the first string literal spanning multiple lines is called out as the likely culprit.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/97001>.

fn main() {
    hello("world);
    // a bunch of code was here
    env("FLAGS", "-help")
    //~^ ERROR prefix `help` is unknown
    //~| ERROR unterminated double quote string
}
