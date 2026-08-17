#![feature(lang_items)]

// Box is expected to be a struct, so this will error.
#[lang = "owned_box"] //~ ERROR the `lang = "owned_box"` attribute cannot be used on statics
static X: u32 = 42;

fn main() {}
