//@ edition: 2021
//@ ignore-windows - we can't force `llvm-cov` to use ANSI escapes on Windows
//@ llvm-cov-flags: --use-color

// Check that column numbers are denoted in bytes, so that they don't cause
// `llvm-cov` to fail or emit malformed output.
//
// Note that when `llvm-cov` prints ^ arrows on a subsequent line, it simply
// inserts one space character for each "column", with no understanding of
// Unicode or character widths. So those arrows will tend to be misaligned
// for non-ASCII source code, regardless of whether column numbers are code
// points or bytes.

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
