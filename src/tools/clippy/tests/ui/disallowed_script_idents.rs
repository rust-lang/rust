#![deny(clippy::disallowed_script_idents)]
#![allow(dead_code)]

fn main() {
    let counter = 10; // OK, latin is allowed.
    let zähler = 10; // OK, it's still latin.

    let счётчик = 10; // Cyrillic is not allowed by default.
    let カウンタ = 10; // Same for japanese.
}
