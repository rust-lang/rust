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

use rustfmt::{WriteMode, run, run_from_stdin};
use rustfmt::config::Config;

use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use getopts::Options;

/// Rustfmt operations.
enum Operation {
    /// Format a file and its child modules.
    Format(PathBuf, WriteMode),
    /// Print the help message.
    Help,
    /// Invalid program input, including reason.
    InvalidInput(String),
    /// No file specified, read from stdin
    Stdin(String, WriteMode),
}

/// Try to find a project file in the input file directory and its parents.
fn lookup_project_file(input_file: &Path) -> io::Result<PathBuf> {
    let mut current = if input_file.is_relative() {
        try!(env::current_dir()).join(input_file)
    } else {
        input_file.to_path_buf()
    };

    // FIXME: We should canonize path to properly handle its parents,
    // but `canonicalize` function is unstable now (recently added API)
    // current = try!(fs::canonicalize(current));

    loop {
        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "Config not found"));
        }
        let config_file = current.join("rustfmt.toml");
        if fs::metadata(&config_file).is_ok() {
            return Ok(config_file);
        }
    }
}

/// Try to find a project file. If it's found, read it.
fn lookup_and_read_project_file(input_file: &Path) -> io::Result<(PathBuf, String)> {
    let path = try!(lookup_project_file(input_file));
    let mut file = try!(File::open(&path));
    let mut toml = String::new();
    try!(file.read_to_string(&mut toml));
    Ok((path, toml))
}

fn execute() -> i32 {
    let mut opts = Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optopt("",
                "write-mode",
                "mode to write in",
                "[replace|overwrite|display|plain|diff|coverage]");

    let operation = determine_operation(&opts, env::args().skip(1));

    match operation {
        Operation::InvalidInput(reason) => {
            print_usage(&opts, &reason);
            1
        }
        Operation::Help => {
            print_usage(&opts, "");
            0
        }
        Operation::Stdin(input, write_mode) => {
            // try to read config from local directory
            let config = match lookup_and_read_project_file(&Path::new(".")) {
                Ok((path, toml)) => {
                    Config::from_toml(&toml)
                }
                Err(_) => Default::default(),
            };

            run_from_stdin(input, write_mode, &config);
            0
        }
        Operation::Format(file, write_mode) => {
            let config = match lookup_and_read_project_file(&file) {
                Ok((path, toml)) => {
                    println!("Using rustfmt config file: {}", path.display());
                    Config::from_toml(&toml)
                }
                Err(_) => Default::default(),
            };

            run(&file, write_mode, &config);
            0
        }
    }
}

fn main() {
    let _ = env_logger::init();
    let exit_code = execute();

    // Make sure standard output is flushed before we exit.
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

fn determine_operation<I>(opts: &Options, args: I) -> Operation
    where I: Iterator<Item = String>
{
    let matches = match opts.parse(args) {
        Ok(m) => m,
        Err(e) => return Operation::InvalidInput(e.to_string()),
    };

    if matches.opt_present("h") {
        return Operation::Help;
    }

    // if no file argument is supplied, read from stdin
    if matches.free.len() != 1 {

        // make sure the write mode is plain or not set
        // (the other options would require a file)
        match matches.opt_str("write-mode") {
            Some(mode) => {
                match mode.parse() {
                    Ok(WriteMode::Plain) => (),
                    _ => return Operation::InvalidInput("Using stdin requires write-mode to be \
                                                         plain"
                                                            .into()),
                }
            }
            _ => (),
        }

        let mut buffer = String::new();
        match io::stdin().read_to_string(&mut buffer) {
            Ok(..) => (),
            Err(e) => return Operation::InvalidInput(e.to_string()),
        }

        return Operation::Stdin(buffer, WriteMode::Plain);
    }

    let write_mode = match matches.opt_str("write-mode") {
        Some(mode) => {
            match mode.parse() {
                Ok(mode) => mode,
                Err(..) => return Operation::InvalidInput("Unrecognized write mode".into()),
            }
        }
        None => WriteMode::Replace,
    };

    Operation::Format(PathBuf::from(&matches.free[0]), write_mode)
}
