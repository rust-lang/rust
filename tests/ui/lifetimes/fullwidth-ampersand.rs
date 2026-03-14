// Verify that we do not ICE when the user uses a multubyte ampersand.
#![allow(todo_macro_uses)]

fn f(_: &＆()) -> &() { todo!() }
//~^ ERROR unknown start of token: \u{ff06}
//~| ERROR missing lifetime specifier [E0106]

fn main() {}
