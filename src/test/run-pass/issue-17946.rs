// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io;
use std::io::fs::PathExtensions;

fn main() {
    let tmp_dir = io::TempDir::new("").unwrap();
    let tmp_file = tmp_dir.path().join("file");
    io::fs::File::create(&tmp_file).unwrap();
    io::fs::change_file_times(&tmp_file, 10000999, 20000999).unwrap();
    let stat = tmp_file.stat().unwrap();
    assert!(stat.accessed == 10000999, "Expected 10000999, got {}", stat.accessed);
    assert!(stat.modified == 20000999, "Expected 20000999, got {}", stat.modified);
}
