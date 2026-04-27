#![feature(lang_items)]

extern "C" {
    #[lang = "copy"] //~ ERROR E0718
    fn copy(); //~ ERROR E0264
}

fn main() {}
