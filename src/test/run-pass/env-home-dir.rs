// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

#![feature(path)]

use std::env::*;
use std::path::PathBuf;

#[cfg(unix)]
fn main() {
    let oldhome = var("HOME");

    set_var("HOME", "/home/MountainView");
    assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

    remove_var("HOME");
    if cfg!(target_os = "android") {
        assert!(home_dir().is_none());
    } else {
        assert!(home_dir().is_some());
    }
}

#[cfg(windows)]
fn main() {
    let oldhome = var("HOME");
    let olduserprofile = var("USERPROFILE");

    remove_var("HOME");
    remove_var("USERPROFILE");

    assert!(home_dir().is_some());

    set_var("HOME", "/home/MountainView");
    assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

    remove_var("HOME");

    set_var("USERPROFILE", "/home/MountainView");
    assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));

    set_var("HOME", "/home/MountainView");
    set_var("USERPROFILE", "/home/PaloAlto");
    assert_eq!(home_dir(), Some(PathBuf::from("/home/MountainView")));
}
