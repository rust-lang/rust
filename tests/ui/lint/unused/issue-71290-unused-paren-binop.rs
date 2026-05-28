//@ check-pass
// Make sure unused parens lint doesn't emit a false positive.
// See https://github.com/rust-lang/rust/issues/71290 for details.
#![deny(unused_parens)]

fn x() -> u8 {
    ({ 0 }) + 1
}

fn y() -> u8 {
    ({ 0 } + 1)
}

pub fn foo(a: bool, b: bool) -> u8 {
    (if a { 1 } else { 0 } + if b { 1 } else { 0 })
}

pub fn bar() -> u8 {
    // Make sure nested expressions are handled correctly as well
    ({ 0 } + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)
}

fn main() {}
