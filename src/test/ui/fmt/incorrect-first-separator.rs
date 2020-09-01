// Allows to track issue #75492:
// https://github.com/rust-lang/rust/issues/75492

use std::iter;

fn main() {
    format!("A number: {}". iter::once(42).next().unwrap());
    //~^ ERROR expected token: `,`

    // Other kind of types are also checked:

    format!("A number: {}" / iter::once(42).next().unwrap());
    //~^ ERROR expected token: `,`

    format!("A number: {}"; iter::once(42).next().unwrap());
    //~^ ERROR expected token: `,`

    // Note: this character is an COMBINING COMMA BELOW unicode char
    format!("A number: {}" ̦ iter::once(42).next().unwrap());
    //~^ ERROR expected token: `,`
    //~^^ ERROR unknown start of token: \u{326}
}
