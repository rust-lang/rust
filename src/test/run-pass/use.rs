#[no_core];
extern mod core;
extern mod zed(name = "core");
extern mod bar(name = "core", vers = "0.5");


use core::str;
use x = zed::str;
mod baz {
    #[legacy_exports];
    use bar::str;
    use x = core::str;
}

fn main() { }