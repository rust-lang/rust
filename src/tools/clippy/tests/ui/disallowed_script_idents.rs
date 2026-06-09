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

fn issue15116() {
    const ÄÖÜ: u8 = 0;
    const _ÄÖÜ: u8 = 0;
    const Ä_ÖÜ: u8 = 0;
    const ÄÖ_Ü: u8 = 0;
    const ÄÖÜ_: u8 = 0;
    let äöüß = 1;
    let _äöüß = 1;
    let ä_öüß = 1;
    let äö_üß = 1;
    let äöü_ß = 1;
    let äöüß_ = 1;
}
