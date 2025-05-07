#![warn(clippy::unnecessary_debug_formatting)]
#![allow(clippy::uninlined_format_args)]

use std::ffi::{OsStr, OsString};

fn main() {
    let os_str = OsStr::new("abc");
    let os_string = os_str.to_os_string();

    // negative tests
    println!("{}", os_str.display());
    println!("{}", os_string.display());

    // positive tests
    println!("{:?}", os_str); //~ unnecessary_debug_formatting
    println!("{:?}", os_string); //~ unnecessary_debug_formatting

    println!("{os_str:?}"); //~ unnecessary_debug_formatting
    println!("{os_string:?}"); //~ unnecessary_debug_formatting

    let _: String = format!("{:?}", os_str); //~ unnecessary_debug_formatting
    let _: String = format!("{:?}", os_string); //~ unnecessary_debug_formatting
}
