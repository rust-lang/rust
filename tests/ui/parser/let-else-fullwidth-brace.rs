#![allow(irrefutable_let_patterns)]

fn main() {
    let x = {1｝ else { return; };
    //~^ ERROR unknown start of token: \u{ff5d}
    //~| ERROR right curly brace `}` before `else` in a `let...else` statement not allowed
}
