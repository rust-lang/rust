//@ run-pass
// Test that the Rust lexer accepts vertical tab (\x0B) as valid whitespace
// between tokens. Vertical tab is part of Unicode Pattern_White_Space, which
// the Rust language specification uses to define whitespace tokens.
// See: https://unicode.org/reports/tr31/#Pattern_White_Space
//
// The space between "let" and "_" below is a vertical tab character (\x0B),
// not a regular space.

fn main() {
    let_ = 1;
}
