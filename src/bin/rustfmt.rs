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

#[macro_use]
extern crate log;
extern crate rustfmt;
extern crate toml;
extern crate env_logger;
extern crate getopts;

use rustfmt::{WriteMode, run};
use rustfmt::config::Config;

use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::PathBuf;

use getopts::Options;

// Try to find a project file in the current directory and its parents.
fn lookup_project_file() -> io::Result<PathBuf> {
    let mut current = try!(env::current_dir());
    loop {
        let config_file = current.join("rustfmt.toml");
        if fs::metadata(&config_file).is_ok() {
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
    let (file, write_mode) = match determine_params(std::env::args().skip(1)) {
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

    run(&file, write_mode, &config);
    0
}

fn main() {
    use std::io::Write;
    let _ = env_logger::init();

    let exit_code = execute();
    // Make sure standard output is flushed before we exit
    std::io::stdout().flush().unwrap();
    // Exit with given exit code.
    //
    // NOTE: This immediately terminates the process without doing any cleanup,
    // so make sure to finish all necessary cleanup before this is called.
    std::process::exit(exit_code);
}

fn print_usage(opts: &Options, reason: &str) {
    let reason = format!("{}\nusage: {} [options] <file>",
                         reason,
                         env::current_exe().unwrap().display());
    println!("{}", opts.usage(&reason));
    Config::print_docs();
}

fn determine_params<I>(args: I) -> Option<(PathBuf, WriteMode)>
    where I: Iterator<Item = String>
{
    let mut opts = Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optopt("",
                "write-mode",
                "mode to write in",
                "[replace|overwrite|display|diff]");
    let matches = match opts.parse(args) {
        Ok(m) => m,
        Err(e) => {
            print_usage(&opts, &e.to_string());
            return None;
        }
    };

    if matches.opt_present("h") {
        print_usage(&opts, "");
    }

    let write_mode = match matches.opt_str("write-mode") {
        Some(mode) => {
            match mode.parse() {
                Ok(mode) => mode,
                Err(..) => {
                    print_usage(&opts, "Unrecognized write mode");
                    return None;
                }
            }
        }
        None => WriteMode::Replace,
    };

    if matches.free.len() != 1 {
        print_usage(&opts, "Please provide one file to format");
        return None;
    }

    Some((PathBuf::from(&matches.free[0]), write_mode))
}
