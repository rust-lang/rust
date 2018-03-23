// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::path::Path;
use std::fs::File;
use std::io::{Read, Write};

fn main() {
    let mut dst = env::current_exe().unwrap();
    dst.pop();
    dst.push("linker-arguments1");
    if dst.exists() {
        dst.pop();
        dst.push("linker-arguments2");
        assert!(!dst.exists());
    }

    let mut out = String::new();
    for arg in env::args().skip(1) {
        let path = Path::new(&arg);
        if !path.is_file() {
            out.push_str(&arg);
            out.push_str("\n");
            continue
        }

        let mut contents = Vec::new();
        File::open(path).unwrap().read_to_end(&mut contents).unwrap();

        out.push_str(&format!("{}: {}\n", arg, hash(&contents)));
    }

    File::create(dst).unwrap().write_all(out.as_bytes()).unwrap();
}

// fnv hash for now
fn hash(contents: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325;

    for byte in contents {
        hash = hash ^ (*byte as u64);
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
}
