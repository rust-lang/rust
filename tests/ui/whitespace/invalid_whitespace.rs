// This test ensures that the Rust lexer rejects invalid whitespace
// characters such as ZERO WIDTH SPACE.

//@ check-fail

fn main() {
    let x = 5;
    let y = 10;

    let a=​x + y;
    //~^ ERROR unknown start of token
    //~| HELP invisible characters like
}
