#![crate_type = "lib"]

use std::path::{Path, PathBuf};

#[cfg(target_os = "linux")]
fn main() -> () {
    let foo = true && false || true;
    let _: *const () = 0;
    let _ = &foo;
    let _ = &&foo;
    let _ = *foo;
    mac!(foo, &mut bar);
    assert!(self.length < N && index <= self.length);
    ::std::env::var("gateau").is_ok();
    #[rustfmt::skip]
    let s:std::path::PathBuf = std::path::PathBuf::new();
    let mut s = String::new();

    match &s {
        ref mut x => {}
    }
}

macro_rules! bar {
    ($foo:tt) => {};
}
