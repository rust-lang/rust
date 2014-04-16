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

// creates a file with `fn main() { <random ident> }` and checks the
// compiler emits a span of the appropriate length (for the
// "unresolved name" message); currently just using the number of code
// points, but should be the number of graphemes (FIXME #7043)

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

    let main_file = tmpdir.join("span_main.rs");
    let main_file_str = main_file.as_str().unwrap();

    for _ in range(0, 100) {
        let n = task_rng().gen_range(3u, 20);

        {
            let _ = write!(&mut File::create(&main_file).unwrap(),
                           r"\#![feature(non_ascii_idents)] fn main() \{ {} \}",
                           // random string of length n
                           range(0, n).map(|_| random_char()).collect::<~str>());
        }

        // rustc is passed to us with --out-dir and -L etc., so we
        // can't exec it directly
        let result = Process::output("sh", ["-c".to_owned(), rustc + " " + main_file_str]).unwrap();

        let err = str::from_utf8_lossy(result.error.as_slice());

        // the span should end the line (e.g no extra ~'s)
        let expected_span = "^" + "~".repeat(n - 1) + "\n";
        assert!(err.as_slice().contains(expected_span));
    }
}
