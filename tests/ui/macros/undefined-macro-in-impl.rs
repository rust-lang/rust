//! regression test for https://github.com/rust-lang/rust/issues/19734
fn main() {}

struct Type;

impl Type {
    undef!();
    //~^ ERROR cannot find macro `undef` in this scope
}
