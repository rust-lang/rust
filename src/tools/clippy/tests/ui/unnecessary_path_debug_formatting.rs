#![warn(clippy::unnecessary_debug_formatting)]
#![allow(clippy::uninlined_format_args)]

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

    // positive tests
    println!("{:?}", os_str); //~ unnecessary_debug_formatting
    println!("{:?}", os_string); //~ unnecessary_debug_formatting

    println!("{:?}", path); //~ unnecessary_debug_formatting
    println!("{:?}", path_buf); //~ unnecessary_debug_formatting

    println!("{path:?}"); //~ unnecessary_debug_formatting
    println!("{path_buf:?}"); //~ unnecessary_debug_formatting

    let _: String = format!("{:?}", path); //~ unnecessary_debug_formatting
    let _: String = format!("{:?}", path_buf); //~ unnecessary_debug_formatting

    let deref_path = DerefPath { path };
    println!("{:?}", &*deref_path); //~ unnecessary_debug_formatting
}

#[test]
fn issue_14345() {
    let input = std::path::Path::new("/foo/bar");
    assert!(input.ends_with("baz"), "{input:?}");
}
