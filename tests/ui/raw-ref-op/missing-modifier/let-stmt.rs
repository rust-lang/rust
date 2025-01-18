//! Check that `&raw` that isn't followed by `const` or `mut` produces a
//! helpful error message.
//!
//! Related issue: <https://github.com/rust-lang/rust/issues/133231>.

fn main() {
    let foo = 2;
    let _ = &raw foo;
    //~^ ERROR expected one of
    //~| HELP you might have meant to use a raw reference
    //~| HELP you might have meant to write a field access
}
