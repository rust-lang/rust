#![feature(native_link_modifiers_bundle)]
#![feature(native_link_modifiers_whole_archive)]
#![feature(native_link_modifiers)]

use std::io::Write;

#[link(name = "c_static_lib_with_constructor",
       kind = "static",
       modifiers = "-bundle,+whole-archive")]
extern {}

pub fn hello() {
    print!("native_lib_in_src.");
    std::io::stdout().flush().unwrap();
}
