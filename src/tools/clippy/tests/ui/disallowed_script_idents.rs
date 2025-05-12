#![deny(clippy::disallowed_script_idents)]
#![allow(dead_code)]

fn main() {
    // OK, latin is allowed.
    let counter = 10;
    // OK, it's still latin.
    let zähler = 10;

    // Cyrillic is not allowed by default.
    let счётчик = 10;
    //~^ disallowed_script_idents

    // Same for japanese.
    let カウンタ = 10;
    //~^ disallowed_script_idents
}
