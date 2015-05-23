// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Build script. Just copies default.toml from the src to the target dir.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let in_file = Path::new("src/default.toml");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let profile = env::var("PROFILE").unwrap();
    let mut out_file = PathBuf::new();
    out_file.push(manifest_dir);
    out_file.push("target");
    out_file.push(profile);
    out_file.push("default.toml");

    std::fs::copy(in_file, out_file).unwrap();
}
