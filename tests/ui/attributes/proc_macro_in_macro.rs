//@ proc-macro: bar_derive.rs
//@ check-pass

extern crate bar_derive;
use bar_derive as bar;

macro_rules! call_macro {
    ($text:expr) => {
        #[derive(bar::Bar)]
        #[arg($text)]
        pub struct Foo;
    };
}

call_macro!(1 + 1);

fn main() {}
