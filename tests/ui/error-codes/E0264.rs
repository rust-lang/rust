#![feature(lang_items)]

extern "C" {
    #[lang = "cake"]
    fn cake(); //~ ERROR E0264
}

fn main() {}
