//@ run-pass
//@ proc-macro: macro-only-syntax.rs

extern crate macro_only_syntax;

#[macro_only_syntax::expect_unsafe_mod]
unsafe mod m {
    pub unsafe mod inner;
}

fn main() {}
