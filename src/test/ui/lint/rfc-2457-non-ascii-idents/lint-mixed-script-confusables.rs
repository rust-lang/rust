#![feature(non_ascii_idents)]
#![deny(mixed_script_confusables)]

struct ΑctuallyNotLatin;
//~^ ERROR The usage of Script Group `Greek` in this crate consists solely of

fn main() {
    let v = ΑctuallyNotLatin;
}

mod роре {
//~^ ERROR The usage of Script Group `Cyrillic` in this crate consists solely of
    const エ: &'static str = "アイウ";
    //~^ ERROR The usage of Script Group `Japanese, Katakana` in this crate consists solely of
}
