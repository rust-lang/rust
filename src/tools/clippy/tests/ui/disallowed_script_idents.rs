#![deny(clippy::disallowed_script_idents)]
#![allow(dead_code)]

fn main() {
    // OK, latin is allowed.
    let counter = 10;
    // OK, it's still latin.
    let zähler = 10;

    // Cyrillic is not allowed by default.
    let счётчик = 10;
    //~^ ERROR: identifier `счётчик` has a Unicode script that is not allowed by configura
    // Same for japanese.
    let カウンタ = 10;
    //~^ ERROR: identifier `カウンタ` has a Unicode script that is not allowed by configuratio
}
