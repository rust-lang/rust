// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(path_ext)]
#![feature(rustc_private)]
#![cfg(not(test))]
#![feature(result_expect)]

#[macro_use]
extern crate log;
extern crate rustfmt;
extern crate toml;

use rustfmt::{WriteMode, run};
use rustfmt::config::Config;

use std::env;
use std::fs::{File, PathExt};
use std::io::{self, Read};
use std::path::PathBuf;
use std::str::FromStr;

// Try to find a project file in the current directory and its parents.
fn lookup_project_file() -> io::Result<PathBuf> {
    let mut current = try!(env::current_dir());
    loop {
        let config_file = current.join("rustfmt.toml");
        if config_file.exists() {
            return Ok(config_file);
        } else {
            current = match current.parent() {
                // if the current directory has no parent, we're done searching
                None => return Err(io::Error::new(io::ErrorKind::NotFound, "config not found")),
                Some(path) => path.to_path_buf(),
            };
        }
    }
}

// Try to find a project file. If it's found, read it.
fn lookup_and_read_project_file() -> io::Result<(PathBuf, String)> {
    let path = try!(lookup_project_file());
    let mut file = try!(File::open(&path));
    let mut toml = String::new();
    try!(file.read_to_string(&mut toml));
    Ok((path, toml))
}

fn main() {
    let (args, write_mode) = determine_params(std::env::args());

    let config = match lookup_and_read_project_file() {
        Ok((path, toml)) => {
            println!("Project config file: {}", path.display());
            Config::from_toml(&toml)
        }
        Err(_) => Default::default(),
    };

    run(args, write_mode, Box::new(config));
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
