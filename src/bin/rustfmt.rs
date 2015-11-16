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
use std::process::Command;
use std::path::{Path, PathBuf};

use getopts::{Matches, Options};

/// Rustfmt operations.
enum Operation {
    /// Format files and their child modules.
    Format(Vec<PathBuf>, WriteMode),
    /// Print the help message.
    Help,
    // Print version information
    Version,
    /// Print detailed configuration help.
    ConfigHelp,
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
        let config_file = current.join("rustfmt.toml");
        if fs::metadata(&config_file).is_ok() {
            return Ok(config_file);
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "Config not found"));
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

fn update_config(config: &mut Config, matches: &Matches) {
    config.verbose = matches.opt_present("verbose");
}

fn execute() -> i32 {
    let mut opts = Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optflag("V", "version", "show version information");
    opts.optflag("v", "verbose", "show progress");
    opts.optopt("",
                "write-mode",
                "mode to write in (not usable when piping from stdin)",
                "[replace|overwrite|display|diff|coverage]");

    opts.optflag("",
                 "config-help",
                 "show details of rustfmt configuration options");

    let matches = match opts.parse(env::args().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            print_usage(&opts, &e.to_string());
            return 1;
        }
    };

    let operation = determine_operation(&matches);

    match operation {
        Operation::InvalidInput(reason) => {
            print_usage(&opts, &reason);
            1
        }
        Operation::Help => {
            print_usage(&opts, "");
            0
        }
        Operation::Version => {
            print_version();
            0
        }
        Operation::ConfigHelp => {
            Config::print_docs();
            0
        }
        Operation::Stdin(input, write_mode) => {
            // try to read config from local directory
            let config = match lookup_and_read_project_file(&Path::new(".")) {
                Ok((_, toml)) => {
                    Config::from_toml(&toml)
                }
                Err(_) => Default::default(),
            };

            run_from_stdin(input, write_mode, &config);
            0
        }
        Operation::Format(files, write_mode) => {
            for file in files {
                let mut config = match lookup_and_read_project_file(&file) {
                    Ok((path, toml)) => {
                        println!("Using rustfmt config file {} for {}",
                                 path.display(),
                                 file.display());
                        Config::from_toml(&toml)
                    }
                    Err(_) => Default::default(),
                };

                update_config(&mut config, &matches);
                run(&file, write_mode, &config);
            }
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
    let reason = format!("{}\nusage: {} [options] <file>...",
                         reason,
                         env::current_exe().unwrap().display());
    println!("{}", opts.usage(&reason));
}

fn print_version() {
    let cmd = Command::new("git")
                  .arg("rev-parse")
                  .arg("--short")
                  .arg("HEAD")
                  .output();
    match cmd {
        Ok(output) => print!("{}", String::from_utf8(output.stdout).unwrap()),
        Err(e) => panic!("Unable te get version: {}", e),
    }
}

fn determine_operation(matches: &Matches) -> Operation {
    if matches.opt_present("h") {
        return Operation::Help;
    }

    if matches.opt_present("config-help") {
        return Operation::ConfigHelp;
    }

    if matches.opt_present("version") {
        return Operation::Version;
    }

    // if no file argument is supplied, read from stdin
    if matches.free.len() == 0 {

        let mut buffer = String::new();
        match io::stdin().read_to_string(&mut buffer) {
            Ok(..) => (),
            Err(e) => return Operation::InvalidInput(e.to_string()),
        }

        // WriteMode is always plain for Stdin
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

    let files: Vec<_> = matches.free.iter().map(|a| PathBuf::from(a)).collect();

    Operation::Format(files, write_mode)
}
