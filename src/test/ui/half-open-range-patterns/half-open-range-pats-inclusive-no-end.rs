// Test `X...` and `X..=` range patterns not being allowed syntactically.
// FIXME(Centril): perhaps these should be semantic restrictions.

#![feature(half_open_range_patterns)]

fn main() {}

#[cfg(FALSE)]
fn foo() {
    if let 0... = 1 {} //~ ERROR inclusive range with no end
    if let 0..= = 1 {} //~ ERROR inclusive range with no end
    const X: u8 = 0;
    if let X... = 1 {} //~ ERROR inclusive range with no end
    if let X..= = 1 {} //~ ERROR inclusive range with no end
}
