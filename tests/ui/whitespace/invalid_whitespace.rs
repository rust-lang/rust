// This test ensures that the Rust lexer rejects invalid whitespace
// characters that are not part of Unicode Pattern_White_Space.
//
// Here we use a ZERO WIDTH SPACE (\u{200B}) via escape to avoid
// issues with invisible characters in editors.

//@ check-fail

fn main() {
    let x = 5;
    let y = 10;

    let a\u{200B}= x + y;
    //~^ ERROR unknown start of token
}
