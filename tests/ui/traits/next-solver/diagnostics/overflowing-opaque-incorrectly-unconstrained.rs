//@ compile-flags: -Znext-solver

// FIXME: The diagnostics here are suboptimal.
// See <https://github.com/rust-lang/rust/issues/163784>.

#![feature(type_alias_impl_trait)]
//~^^^^^^ ERROR: overflow evaluating the requirement `X == _`

// TAIT with self-referencing bounds
type X = impl std::ops::Add<Output = X>;

struct Foo;

impl Foo {
    #[define_opaque(X)]
    pub fn new() -> X {
        //~^ ERROR: item does not constrain `X::{opaque#0}`
        0i32
    }
}

fn main() {}
