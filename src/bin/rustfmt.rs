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
#![feature(result_expect)]

extern crate rustfmt;

use rustfmt::{WriteMode, run};
use rustfmt::config::Config;

use std::fs::File;
use std::io::Read;
use std::str::FromStr;

fn main() {
    let mut def_config_file = File::open("default.toml").unwrap_or_else(|e| {
        panic!("Unable to open configuration file [default.toml] {}",e)
    });
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).unwrap();
    let config = Box::new(Config::from_toml(&def_config));
    let (args, write_mode) = determine_params(std::env::args());

    run(args, write_mode, config);

    std::process::exit(0);
}

fn determine_params<I>(args: I) -> (Vec<String>, WriteMode)
    where I: Iterator<Item = String>
{
    let prefix = "--write-mode=";
    let mut write_mode = WriteMode::Replace;

    // The NewFile option currently isn't supported because it requires another
    // parameter, but it can be added later.
    let args = args.filter(|arg| {
        if arg.starts_with(prefix) {
            write_mode = FromStr::from_str(&arg[prefix.len()..]).expect("Unrecognized write mode");
            false
        } else {
            true
        }
    }).collect();

    (args, write_mode)
}
