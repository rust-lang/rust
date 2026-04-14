// This test ensures that the Rust lexer rejects invalid whitespace
// characters that are not part of Unicode Pattern_White_Space.
//
// This uses a ZERO WIDTH SPACE (U+200B), which is not valid Rust whitespace.

//@ check-fail

fn main() {
    let x = 5;
    let y = 10;

    let a​= x + y;
    //~^ ERROR unknown start of token
    //~| HELP invisible characters
}
