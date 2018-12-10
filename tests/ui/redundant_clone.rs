// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::redundant_clone)]

use std::path::Path;
use std::ffi::OsString;

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
    #[allow(clippy::redundant_clone)] let _ = String::new().to_string();

    let tup = (String::from("foo"),);
    let _ = tup.0.clone();

    let tup_ref = &(String::from("foo"),);
    let _s = tup_ref.0.clone(); // this `.clone()` cannot be removed
}

#[derive(Clone)]
struct Alpha;
fn with_branch(a: Alpha, b: bool) -> (Alpha, Alpha) {
    if b {
        (a.clone(), a.clone())
    } else {
        (Alpha, a)
    }
}

struct TypeWithDrop {
    x: String,
}

impl Drop for TypeWithDrop {
    fn drop(&mut self) {}
}

fn cannot_move_from_type_with_drop() -> String {
    let s = TypeWithDrop {
        x: String::new()
    };
    s.x.clone() // removing this `clone()` summons E0509
}
