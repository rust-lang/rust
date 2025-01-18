//! Check that `&raw` that isn't followed by `const` or `mut` produces a
//! helpful error message.
//!
//! Related issue: <https://github.com/rust-lang/rust/issues/133231>.

mod foo {
    pub static A: i32 = 0;
}

fn main() {
    [&raw foo::A; 3];
    //~^ ERROR expected one of
    //~| HELP you might have meant to use a raw reference
}
