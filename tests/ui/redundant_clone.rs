// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::redundant_clone)]

use std::ffi::OsString;
use std::path::Path;

fn main() {
    let _ = ["lorem", "ipsum"].join(" ").to_string();

    let s = String::from("foo");
    let _ = s.clone();

    let s = String::from("foo");
    let _ = s.to_string();

    let s = String::from("foo");
    let _ = s.to_owned();

    let _ = Path::new("/a/b/").join("c").to_owned();

    let _ = Path::new("/a/b/").join("c").to_path_buf();

    let _ = OsString::new().to_owned();

    let _ = OsString::new().to_os_string();

    // Check that lint level works
    #[allow(clippy::redundant_clone)]
    let _ = String::new().to_string();
}

#[derive(Clone)]
struct Alpha;
fn double(a: Alpha) -> (Alpha, Alpha) {
    if true {
        (a.clone(), a.clone())
    } else {
        (Alpha, a)
    }
}
