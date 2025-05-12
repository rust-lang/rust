//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

use std::path::Path;
use std::ffi::OsStr;
use std::ops::Deref;

fn frob(path: &str) -> impl Deref<Target = Path> + '_ {
    OsStr::new(path).as_ref()
}

fn open_parent<'path>(_path: &'path Path) {
    todo!()
}

fn main() {
    let old_path = frob("hello");

    open_parent(&old_path);
}
