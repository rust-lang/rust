#![feature(lang_items)]

extern "C" {
    #[lang = "copy"] //~ ERROR E0264
    fn copy();
}

fn main() {}
