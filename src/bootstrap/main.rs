// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

extern crate bootstrap;
extern crate build_helper;
extern crate cmake;
extern crate filetime;
extern crate gcc;
extern crate getopts;
extern crate libc;
extern crate num_cpus;
extern crate rustc_serialize;
extern crate toml;
extern crate md5;

use std::env;

use build::{Flags, Config, Build};

mod build;

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let flags = Flags::parse(&args);
    let mut config = Config::parse(&flags.build, flags.config.clone());
    if std::fs::metadata("config.mk").is_ok() {
        config.update_with_config_mk();
    }
    Build::new(flags, config).build();
}
