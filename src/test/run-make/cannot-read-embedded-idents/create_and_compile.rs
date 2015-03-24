// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(old_io, old_path)]

use std::env;
use std::fs::File;
use std::process::Command;
use std::io::Write;
use std::path::Path;

// creates broken.rs, which has the Ident \x00name_0,ctxt_0\x00
// embedded within it, and then attempts to compile broken.rs with the
// provided `rustc`

fn main() {
    let args: Vec<String> = env::args().collect();
    let rustc = &args[1];
    let tmpdir = Path::new(&args[2]);

    let main_file = tmpdir.join("broken.rs");
    let _ = File::create(&main_file).unwrap()
        .write_all(b"pub fn main() {
                   let \x00name_0,ctxt_0\x00 = 3;
                   println!(\"{}\", \x00name_0,ctxt_0\x00);
        }").unwrap();

    // rustc is passed to us with --out-dir and -L etc., so we
    // can't exec it directly
    let result = Command::new("sh")
        .arg("-c")
        .arg(&format!("{} {}", rustc, main_file.display()))
        .output().unwrap();
    let err = String::from_utf8_lossy(&result.stderr);

    // positive test so that this test will be updated when the
    // compiler changes.
    assert!(err.contains("unknown start of token"))
}
