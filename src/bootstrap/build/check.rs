// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::process::Command;

use build::{Build, Compiler};

pub fn linkcheck(build: &Build, stage: u32, host: &str) {
    println!("Linkcheck stage{} ({})", stage, host);
    let compiler = Compiler::new(stage, host);
    let linkchecker = build.tool(&compiler, "linkchecker");
    build.run(Command::new(&linkchecker)
                     .arg(build.out.join(host).join("doc")));
}
