// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

extern mod std;
use core::io::WriterUtil;
use std::tempfile;

pub fn main() {
    let dir = tempfile::mkdtemp(&Path("."), "").unwrap();
    let path = dir.with_filename("file");

    {
        match io::file_writer(&path, [io::Create, io::Truncate]) {
            Err(ref e) => fail!(e.clone()),
            Ok(f) => {
                for uint::range(0, 1000) |_i| {
                    f.write_u8(0);
                }
            }
        }
    }

    assert!(path.exists());
    assert!(path.get_size() == Some(1000));

    os::remove_dir(&dir);
}
