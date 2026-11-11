// Allows to track issue #75492:
// https://github.com/rust-lang/rust/issues/75492

use std::iter;

fn main() {
    format!("A number: {}". iter::once(42).next().unwrap());
    //~^ ERROR expected `,`, found `.`

    // Other kind of types are also checked:

    format!("A number: {}" / iter::once(42).next().unwrap());
    //~^ ERROR expected `,`, found `/`

    format!("A number: {}"; iter::once(42).next().unwrap());
    //~^ ERROR expected `,`, found `;`

    // Note: this character is an COMBINING COMMA BELOW unicode char
    format!("A number: {}" ̦ iter::once(42).next().unwrap());
    //~^ ERROR expected `,`, found `iter`
    //~^^ ERROR unknown start of token: \u{326}

    // Here recovery is tested.
    // If the `compile_error!` is emitted, then the parser is able to recover
    // from the incorrect first separator.
    format!("{}". compile_error!("fail"));
    //~^ ERROR expected `,`, found `.`
    //~^^ ERROR fail
}
