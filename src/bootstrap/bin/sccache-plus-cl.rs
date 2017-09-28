// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate cc;

use std::env;
use std::process::{self, Command};

fn main() {
    let target = env::var("SCCACHE_TARGET").unwrap();
    // Locate the actual compiler that we're invoking
    env::remove_var("CC");
    env::remove_var("CXX");
    let mut cfg = cc::Build::new();
    cfg.cargo_metadata(false)
       .out_dir("/")
       .target(&target)
       .host(&target)
       .opt_level(0)
       .warnings(false)
       .debug(false);
    let compiler = cfg.get_compiler();

    // Invoke sccache with said compiler
    let sccache_path = env::var_os("SCCACHE_PATH").unwrap();
    let mut cmd = Command::new(&sccache_path);
    cmd.arg(compiler.path());
    for &(ref k, ref v) in compiler.env() {
        cmd.env(k, v);
    }
    for arg in env::args().skip(1) {
        cmd.arg(arg);
    }

    let status = cmd.status().expect("failed to spawn");
    process::exit(status.code().unwrap_or(2))
}
