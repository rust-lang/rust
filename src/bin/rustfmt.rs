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

#[macro_use]
extern crate log;
extern crate rustfmt;
extern crate toml;

use rustfmt::{WriteMode, run};
use rustfmt::config::Config;
use rustfmt::config::ConfigHelpVariantTypes;

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

fn execute() -> i32 {
    let (args, write_mode) = match determine_params(std::env::args()) {
        Some(params) => params,
        None => return 1,
    };

    let config = match lookup_and_read_project_file() {
        Ok((path, toml)) => {
            println!("Project config file: {}", path.display());
            Config::from_toml(&toml)
        }
        Err(_) => Default::default(),
    };

    run(args, write_mode, Box::new(config));
    0
}

fn main() {
    use std::io::Write;
    let exit_code = execute();
    // Make sure standard output is flushed before we exit
    std::io::stdout().flush().unwrap();
    // Exit with given exit code.
    //
    // NOTE: This immediately terminates the process without doing any cleanup,
    // so make sure to finish all necessary cleanup before this is called.
    std::process::exit(exit_code);
}

fn print_usage<S: Into<String>>(reason: S) {
    println!("{}\n\r usage: rustfmt [-h Help] [--write-mode=[replace|overwrite|display|diff]] \
              <file_name>",
             reason.into());

    for option in Config::get_docs() {
        let variants = option.variant_names();
        let variant_names: String = match *variants {
            ConfigHelpVariantTypes::UsizeConfig => "<unsigned integer>".into(),
            ConfigHelpVariantTypes::BoolConfig => "<boolean>".into(),
            ConfigHelpVariantTypes::EnumConfig(ref variants) => variants.join(", "),
        };
        println!("{}, {}, Possible values: {}", option.option_name(), option.doc_string(), variant_names);
    }
}

fn determine_params<I>(args: I) -> Option<(Vec<String>, WriteMode)>
    where I: Iterator<Item = String>
{
    let arg_prefix = "-";
    let write_mode_prefix = "--write-mode=";
    let help_mode = "-h";
    let long_help_mode = "--help";
    let mut write_mode = WriteMode::Replace;
    let mut rustc_args = Vec::new();

    // The NewFile option currently isn't supported because it requires another
    // parameter, but it can be added later.
    for arg in args {
        if arg.starts_with(write_mode_prefix) {
            match FromStr::from_str(&arg[write_mode_prefix.len()..]) {
                Ok(mode) => write_mode = mode,
                Err(_) => {
                    print_usage("Unrecognized write mode");
                    return None;
                }
            }
        } else if arg.starts_with(help_mode) || arg.starts_with(long_help_mode) {
            print_usage("");
            return None;
        } else if arg.starts_with(arg_prefix) {
            print_usage("Invalid argument");
            return None;
        } else {
            // Pass everything else to rustc
            rustc_args.push(arg);
        }
    }

    if rustc_args.len() < 2 {
        print_usage("Please provide a file to be formatted");
        return None;
    }

    Some((rustc_args, write_mode))
}
