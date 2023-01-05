#![deny(mixed_script_confusables)]

struct ΑctuallyNotLatin;
//~^ ERROR the usage of Script Group `Greek` in this crate consists solely of

fn main() {
    let v = ΑctuallyNotLatin;
}

mod роре {
//~^ ERROR the usage of Script Group `Cyrillic` in this crate consists solely of
    const エ: &'static str = "アイウ";
    //~^ ERROR the usage of Script Group `Japanese, Katakana` in this crate consists solely of
}
