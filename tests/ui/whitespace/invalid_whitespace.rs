// This test ensures that the Rust lexer rejects invalid whitespace
// characters such as ZERO WIDTH SPACE.

//@ check-fail

fn main() {
    let x = 5;
    let y = 10;

    let a​= x + y;
    //~^ ERROR unknown start of token: \u{200b}
    //~| HELP invisible characters like '\u{200b}' are not usually visible in text editors
}
