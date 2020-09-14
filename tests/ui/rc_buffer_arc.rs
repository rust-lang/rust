use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

#[warn(clippy::rc_buffer)]
struct S {
    a: Arc<String>,
    b: Arc<PathBuf>,
    c: Arc<Vec<u8>>,
    d: Arc<OsString>,
}

fn main() {}
