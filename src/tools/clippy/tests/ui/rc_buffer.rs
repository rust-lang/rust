#![warn(clippy::rc_buffer)]

use std::cell::RefCell;
use std::ffi::OsString;
use std::path::PathBuf;
use std::rc::Rc;

struct S {
    // triggers lint
    bad1: Rc<String>,
    bad2: Rc<PathBuf>,
    bad3: Rc<Vec<u8>>,
    bad4: Rc<OsString>,
    // does not trigger lint
    good1: Rc<RefCell<String>>,
}

// triggers lint
fn func_bad1(_: Rc<String>) {}
fn func_bad2(_: Rc<PathBuf>) {}
fn func_bad3(_: Rc<Vec<u8>>) {}
fn func_bad4(_: Rc<OsString>) {}
// does not trigger lint
fn func_good1(_: Rc<RefCell<String>>) {}

fn main() {}
