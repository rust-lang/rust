
// This test ensures that the Rust lexer rejects invalid whitespace
// characters that are not part of Unicode Pattern_White_Space.
//
// In this case, we use ZERO WIDTH SPACE (\u{200B}), which should
// not be accepted as valid whitespace by the lexer.

//@ check-fail

fn main() {
    let x = 5;
    let y = 10;

    // The line below contains a ZERO WIDTH SPACE between `a` and `=`
    let a​= x + y;
    //~^ ERROR unknown start of token
}


