use std::ffi::OsString;
use std::path::PathBuf;
use std::rc::Rc;

#[warn(clippy::rc_buffer)]
struct S {
    a: Rc<String>,
    b: Rc<PathBuf>,
    c: Rc<Vec<u8>>,
    d: Rc<OsString>,
}

fn main() {}
