// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporary files and directories

use core::rand::RngUtil;

pub fn mkdtemp(tmpdir: &Path, suffix: &str) -> Option<Path> {
    let r = rand::rng();
    for 1000.times {
        let p = tmpdir.push(r.gen_str(16) + suffix);
        if os::make_dir(&p, 0x1c0) { // 700
            return Some(p);
        }
    }
    None
}

#[test]
fn test_mkdtemp() {
    let p = mkdtemp(&Path("."), "foobar").unwrap();
    os::remove_dir(&p);
    assert!(str::ends_with(p.to_str(), "foobar"));
}
