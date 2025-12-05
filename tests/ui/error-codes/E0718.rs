#![feature(lang_items)]

// Box is expected to be a struct, so this will error.
#[lang = "owned_box"] //~ ERROR lang item must be applied to a struct
static X: u32 = 42;

fn main() {}
