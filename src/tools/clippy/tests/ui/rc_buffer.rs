#![warn(clippy::rc_buffer)]
#![allow(dead_code, unused_imports)]

use std::cell::RefCell;
use std::ffi::OsString;
use std::path::PathBuf;
use std::rc::Rc;

struct S {
    // triggers lint
    bad1: Rc<String>,
    //~^ rc_buffer
    bad2: Rc<PathBuf>,
    //~^ rc_buffer
    bad3: Rc<Vec<u8>>,
    //~^ rc_buffer
    bad4: Rc<OsString>,
    //~^ rc_buffer
    // does not trigger lint
    good1: Rc<RefCell<String>>,
}

// triggers lint
fn func_bad1(_: Rc<String>) {}
//~^ rc_buffer
fn func_bad2(_: Rc<PathBuf>) {}
//~^ rc_buffer
fn func_bad3(_: Rc<Vec<u8>>) {}
//~^ rc_buffer
fn func_bad4(_: Rc<OsString>) {}
//~^ rc_buffer
// does not trigger lint
fn func_good1(_: Rc<RefCell<String>>) {}

fn main() {}
