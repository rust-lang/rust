#![allow(unused)]
fn main() {
    let _foo = b'hello\0';
    //~^ ERROR character literal may only contain one codepoint
    //~| HELP if you meant to write a byte string literal, use double quotes
    let _bar = 'hello';
    //~^ ERROR character literal may only contain one codepoint
    //~| HELP if you meant to write a `str` literal, use double quotes
}
