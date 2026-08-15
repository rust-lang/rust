use std::io::Write;

#[link(
    name = "c_static_lib_with_constructor",
    kind = "static",
    modifiers = "-bundle,+whole-archive"
)]
extern "C" {}

pub fn hello() {
    print!("native_lib_in_src.");
    std::io::stdout().flush().unwrap();
}
