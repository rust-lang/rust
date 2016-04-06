// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{Build, Compiler};

pub fn linkcheck(build: &Build, stage: u32, host: &str) {
    println!("Linkcheck stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    build.run(build.tool_cmd(&compiler, "linkchecker")
                   .arg(build.out.join(host).join("doc")));
}

pub fn cargotest(build: &Build, stage: u32, host: &str) {

    let ref compiler = Compiler::new(stage, host);

    // Configure PATH to find the right rustc. NB. we have to use PATH
    // and not RUSTC because the Cargo test suite has tests that will
    // fail if rustc is not spelled `rustc`.
    let path = build.sysroot(compiler).join("bin");
    let old_path = ::std::env::var("PATH").expect("");
    let sep = if cfg!(windows) { ";" } else {":" };
    let ref newpath = format!("{}{}{}", path.display(), sep, old_path);

    build.run(build.tool_cmd(compiler, "cargotest")
              .env("PATH", newpath)
              .arg(&build.cargo));
}
