#![warn(clippy::unnecessary_debug_formatting)]

use std::ffi::{OsStr, OsString};
use std::ops::Deref;
use std::path::{Path, PathBuf};

struct DerefPath<'a> {
    path: &'a Path,
}

impl Deref for DerefPath<'_> {
    type Target = Path;
    fn deref(&self) -> &Self::Target {
        self.path
    }
}

fn main() {
    let path = Path::new("/a/b/c");
    let path_buf = path.to_path_buf();
    let os_str = OsStr::new("abc");
    let os_string = os_str.to_os_string();

    // negative tests
    println!("{}", path.display());
    println!("{}", path_buf.display());

    // should not fire because feature `os_str_display` is not enabled
    println!("{:?}", os_str);
    println!("{:?}", os_string);

    // positive tests
    println!("{:?}", path); //~ unnecessary_debug_formatting
    println!("{:?}", path_buf); //~ unnecessary_debug_formatting

    println!("{path:?}"); //~ unnecessary_debug_formatting
    println!("{path_buf:?}"); //~ unnecessary_debug_formatting

    let _: String = format!("{:?}", path); //~ unnecessary_debug_formatting
    let _: String = format!("{:?}", path_buf); //~ unnecessary_debug_formatting

    let deref_path = DerefPath { path };
    println!("{:?}", &*deref_path); //~ unnecessary_debug_formatting
}
