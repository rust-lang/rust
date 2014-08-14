// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::os;
use std::io::{File, Command};

// creates broken.rs, which has the Ident \x00name_0,ctxt_0\x00
// embedded within it, and then attempts to compile broken.rs with the
// provided `rustc`

fn main() {
    let args = os::args();
    let rustc = args[1].as_slice();
    let tmpdir = Path::new(args[2].as_slice());

    let main_file = tmpdir.join("broken.rs");
    let _ = File::create(&main_file).unwrap()
        .write_str("pub fn main() {
                   let \x00name_0,ctxt_0\x00 = 3i;
                   println!(\"{}\", \x00name_0,ctxt_0\x00);
        }");

    // rustc is passed to us with --out-dir and -L etc., so we
    // can't exec it directly
    let result = Command::new("sh")
        .arg("-c")
        .arg(format!("{} {}",
                     rustc,
                     main_file.as_str()
                     .unwrap()).as_slice())
        .output().unwrap();
    let err = String::from_utf8_lossy(result.error.as_slice());

    // positive test so that this test will be updated when the
    // compiler changes.
    assert!(err.as_slice().contains("unknown start of token"))
}
