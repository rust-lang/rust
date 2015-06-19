// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(not(test))]

extern crate rustfmt;

use rustfmt::{WriteMode, run};

use std::fs::File;
use std::io::Read;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mut def_config_file = File::open("default.toml").unwrap();
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).unwrap();

    run(args, WriteMode::Overwrite, &def_config);

    std::process::exit(0);
}
