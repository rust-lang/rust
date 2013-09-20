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


use std::os;
use std::rand::Rng;
use std::rand;

/// Attempts to make a temporary directory inside of `tmpdir` whose name will
/// have the suffix `suffix`. If no directory can be created, None is returned.
pub fn mkdtemp(tmpdir: &Path, suffix: &str) -> Option<Path> {
    let mut r = rand::rng();
    for _ in range(0u, 1000) {
        let p = tmpdir.push(r.gen_ascii_str(16) + suffix);
        if os::make_dir(&p, 0x1c0) { // 700
            return Some(p);
        }
    }
    None
}

// the tests for this module need to change the path using change_dir,
// and this doesn't play nicely with other tests so these unit tests are located
// in src/test/run-pass/tempfile.rs
