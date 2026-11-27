// Regression test for <https://github.com/rust-lang/rust/issues/140612>.
//@ proc-macro: derive_macro_with_helper.rs
//@ edition: 2018
//@ check-pass

macro_rules! expand {
    ($text:expr) => {
        #[derive(derive_macro_with_helper::Derive)]
        // This inert attr is completely valid because it follows the grammar
        // `#` `[` SimplePath DelimitedTokenStream `]`.
        // However, we used to incorrectly delay a bug here and ICE when trying to parse `$text` as
        // the inside of a "meta item list" which may only begin with literals or paths.
        #[arg($text)]
        pub struct Foo;
    };
}

expand!(1 + 1);

fn main() {}
