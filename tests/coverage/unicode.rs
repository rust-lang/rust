//@ edition: 2021
//@ llvm-cov-flags: --format=html

// Check that column numbers are denoted in bytes, so that they don't cause
// `llvm-cov` to fail or emit malformed output.
//
// Regression test for <https://github.com/rust-lang/rust/pull/119033>.

fn main() {
    for _İ in 'А'..='Я' { /* Я */ }

    if 申し訳ございません() && 申し訳ございません() {
        println!("true");
    }

    サビ();
}

fn 申し訳ございません() -> bool {
    std::hint::black_box(false)
}

macro_rules! macro_that_defines_a_function {
    (fn $名:ident () $体:tt) => {
        fn $名 () $体 fn 他 () {}
    }
}

macro_that_defines_a_function! {
    fn サビ() {}
}
