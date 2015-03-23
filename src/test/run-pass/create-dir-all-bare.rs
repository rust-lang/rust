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
use std::fs::{self, TempDir};

fn main() {
    let td = TempDir::new("create-dir-all-bare").unwrap();
    env::set_current_dir(td.path()).unwrap();
    fs::create_dir_all("create-dir-all-bare").unwrap();
}
