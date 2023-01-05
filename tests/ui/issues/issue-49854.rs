// run-pass
use std::ffi::OsString;

fn main() {
    let os_str = OsString::from("Hello Rust!");

    assert_eq!(os_str, "Hello Rust!");
    assert_eq!("Hello Rust!", os_str);
}
