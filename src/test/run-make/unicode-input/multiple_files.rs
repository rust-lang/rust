// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rand;
use rand::{task_rng, Rng};

use std::{char, os, str};
use std::io::{File, Process};

// creates unicode_input_multiple_files_{main,chars}.rs, where the
// former imports the latter. `_chars` just contains an indentifier
// made up of random characters, because will emit an error message
// about the ident being in the wrong place, with a span (and creating
// this span used to upset the compiler).

fn random_char() -> char {
    let mut rng = task_rng();
    // a subset of the XID_start unicode table (ensuring that the
    // compiler doesn't fail with an "unrecognised token" error)
    let (lo, hi): (u32, u32) = match rng.gen_range(1, 4 + 1) {
        1 => (0x41, 0x5a),
        2 => (0xf8, 0x1ba),
        3 => (0x1401, 0x166c),
        _ => (0x10400, 0x1044f)
    };

    char::from_u32(rng.gen_range(lo, hi + 1)).unwrap()
}

fn main() {
    let args = os::args();
    let rustc = args[1].as_slice();
    let tmpdir = Path::new(args[2].as_slice());

    let main_file = tmpdir.join("unicode_input_multiple_files_main.rs");
    let main_file_str = main_file.as_str().unwrap();
    {
        let _ = File::create(&main_file).unwrap()
            .write_str("mod unicode_input_multiple_files_chars;");
    }

    for _ in range(0, 100) {
        {
            let randoms = tmpdir.join("unicode_input_multiple_files_chars.rs");
            let mut w = File::create(&randoms).unwrap();
            for _ in range(0, 30) {
                let _ = w.write_char(random_char());
            }
        }

        // rustc is passed to us with --out-dir and -L etc., so we
        // can't exec it directly
        let result = Process::output("sh", ["-c".to_owned(), rustc + " " + main_file_str]).unwrap();
        let err = str::from_utf8_lossy(result.error.as_slice());

        // positive test so that this test will be updated when the
        // compiler changes.
        assert!(err.as_slice().contains("expected item but found"))
    }
}
