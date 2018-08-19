// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let path = PathBuf::from(env::var_os("RUST_TEST_TMPDIR").unwrap());
    env::set_current_dir(&path).unwrap();
    fs::create_dir_all("create-dir-all-bare").unwrap();
}
