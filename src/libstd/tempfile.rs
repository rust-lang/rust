// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporary files and directories

#[forbid(deprecated_mode)];

use core::option;
use option::{None, Some};

pub fn mkdtemp(tmpdir: &Path, suffix: &str) -> Option<Path> {
    let r = rand::Rng();
    let mut i = 0u;
    while (i < 1000u) {
        let p = tmpdir.push(r.gen_str(16u) +
                            str::from_slice(suffix));
        if os::make_dir(&p, 0x1c0i32) {  // FIXME: u+rwx (#2349)
            return Some(p);
        }
        i += 1u;
    }
    return None;
}

#[test]
fn test_mkdtemp() {
    let r = mkdtemp(&Path("."), "foobar");
    match r {
        Some(ref p) => {
            os::remove_dir(p);
            assert(str::ends_with(p.to_str(), "foobar"));
        }
        _ => assert(false)
    }
}
