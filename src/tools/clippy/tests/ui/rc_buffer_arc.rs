#![warn(clippy::rc_buffer)]
#![allow(dead_code, unused_imports)]

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct S {
    // triggers lint
    bad1: Arc<String>,
    //~^ rc_buffer
    bad2: Arc<PathBuf>,
    //~^ rc_buffer
    bad3: Arc<Vec<u8>>,
    //~^ rc_buffer
    bad4: Arc<OsString>,
    //~^ rc_buffer
    // does not trigger lint
    good1: Arc<Mutex<String>>,
}

// triggers lint
fn func_bad1(_: Arc<String>) {}
//~^ rc_buffer
fn func_bad2(_: Arc<PathBuf>) {}
//~^ rc_buffer
fn func_bad3(_: Arc<Vec<u8>>) {}
//~^ rc_buffer
fn func_bad4(_: Arc<OsString>) {}
//~^ rc_buffer
// does not trigger lint
fn func_good1(_: Arc<Mutex<String>>) {}

fn main() {}
