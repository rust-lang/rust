// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The sheer existence of this file is an awful hack. See the comments in
// `src/bootstrap/native.rs` for why this is needed when compiling LLD.

use std::env;
use std::process::{self, Stdio, Command};
use std::io::{self, Write};

fn main() {
    let real_llvm_config = env::var_os("LLVM_CONFIG_REAL").unwrap();
    let mut cmd = Command::new(real_llvm_config);
    cmd.args(env::args().skip(1)).stderr(Stdio::piped());
    let output = cmd.output().expect("failed to spawn llvm-config");
    let stdout = String::from_utf8_lossy(&output.stdout);
    print!("{}", stdout.replace("\\", "/"));
    io::stdout().flush().unwrap();
    process::exit(output.status.code().unwrap_or(1));
}
