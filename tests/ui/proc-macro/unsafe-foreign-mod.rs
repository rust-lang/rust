//@ run-pass
//@ proc-macro: macro-only-syntax.rs

extern crate macro_only_syntax;

#[macro_only_syntax::expect_unsafe_foreign_mod]
unsafe extern {
    type T;
}

#[macro_only_syntax::expect_unsafe_extern_cpp_mod]
unsafe extern "C++" {}

fn main() {}
