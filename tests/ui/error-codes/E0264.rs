#![feature(lang_items)]

extern "C" {
    #[lang = "copy"]
    fn copy(); //~ ERROR E0264
}

fn main() {}
