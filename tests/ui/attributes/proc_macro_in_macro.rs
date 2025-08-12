// Regression test for <https://github.com/rust-lang/rust/issues/140612>.
//@ proc-macro: derive_macro_with_helper.rs
//@ edition: 2018
//@ check-pass

macro_rules! call_macro {
    ($text:expr) => {
        #[derive(derive_macro_with_helper::Derive)]
        #[arg($text)]
        pub struct Foo;
    };
}

call_macro!(1 + 1);

fn main() {}
