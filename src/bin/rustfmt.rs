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


extern crate log;
extern crate rustfmt;
extern crate toml;
extern crate env_logger;
extern crate getopts;

use rustfmt::{run, Input, Summary};
use rustfmt::config::{Config, WriteMode};

use std::{env, error};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use getopts::{Matches, Options};

type FmtError = Box<error::Error + Send + Sync>;
type FmtResult<T> = std::result::Result<T, FmtError>;

/// Rustfmt operations.
enum Operation {
    /// Format files and their child modules.
    Format {
        files: Vec<PathBuf>,
        config_path: Option<PathBuf>,
    },
    /// Print the help message.
    Help,
    // Print version information
    Version,
    /// Print detailed configuration help.
    ConfigHelp,
    /// No file specified, read from stdin
    Stdin {
        input: String,
        config_path: Option<PathBuf>,
    },
}

/// Parsed command line options.
#[derive(Clone, Debug, Default)]
struct CliOptions {
    skip_children: bool,
    verbose: bool,
    write_mode: Option<WriteMode>,
}

impl CliOptions {
    fn from_matches(matches: &Matches) -> FmtResult<CliOptions> {
        let mut options = CliOptions::default();
        options.skip_children = matches.opt_present("skip-children");
        options.verbose = matches.opt_present("verbose");

        if let Some(ref write_mode) = matches.opt_str("write-mode") {
            if let Ok(write_mode) = WriteMode::from_str(write_mode) {
                options.write_mode = Some(write_mode);
            } else {
                return Err(FmtError::from(format!("Invalid write-mode: {}", write_mode)));
            }
        }

        Ok(options)
    }

    fn apply_to(&self, config: &mut Config) {
        config.skip_children = self.skip_children;
        config.verbose = self.verbose;
        if let Some(write_mode) = self.write_mode {
            config.write_mode = write_mode;
        }
    }
}

/// Try to find a project file in the given directory and its parents. Returns the path of a the
/// nearest project file if one exists, or `None` if no project file was found.
fn lookup_project_file(dir: &Path) -> FmtResult<Option<PathBuf>> {
    let mut current = if dir.is_relative() {
        try!(env::current_dir()).join(dir)
    } else {
        dir.to_path_buf()
    };

    current = try!(fs::canonicalize(current));

    loop {
        let config_file = current.join("rustfmt.toml");
        match fs::metadata(&config_file) {
            // Only return if it's a file to handle the unlikely situation of a directory named
            // `rustfmt.toml`.
            Ok(ref md) if md.is_file() => return Ok(Some(config_file)),
            // Return the error if it's something other than `NotFound`; otherwise we didn't find
            // the project file yet, and continue searching.
            Err(e) => {
                if e.kind() != ErrorKind::NotFound {
                    return Err(FmtError::from(e));
                }
            }
            _ => {}
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return Ok(None);
        }
    }
}

/// Resolve the config for input in `dir`.
///
/// Returns the `Config` to use, and the path of the project file if there was
/// one.
fn resolve_config(dir: &Path) -> FmtResult<(Config, Option<PathBuf>)> {
    let path = try!(lookup_project_file(dir));
    if path.is_none() {
        return Ok((Config::default(), None));
    }
    let path = path.unwrap();
    let mut file = try!(File::open(&path));
    let mut toml = String::new();
    try!(file.read_to_string(&mut toml));
    Ok((Config::from_toml(&toml), Some(path)))
}

/// read the given config file path recursively if present else read the project file path
fn match_cli_path_or_file(config_path: Option<PathBuf>,
                          input_file: &Path)
                          -> FmtResult<(Config, Option<PathBuf>)> {

    if let Some(config_file) = config_path {
        let (toml, path) = try!(resolve_config(config_file.as_ref()));
        if path.is_some() {
            return Ok((toml, path));
        }
    }
    resolve_config(input_file)
}

fn make_opts() -> Options {
    let mut opts = Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optflag("V", "version", "show version information");
    opts.optflag("v", "verbose", "show progress");
    opts.optopt("",
                "write-mode",
                "mode to write in (not usable when piping from stdin)",
                "[replace|overwrite|display|diff|coverage|checkstyle]");
    opts.optflag("", "skip-children", "don't reformat child modules");

    opts.optflag("",
                 "config-help",
                 "show details of rustfmt configuration options");
    opts.optopt("",
                "config-path",
                "Recursively searches the given path for the rustfmt.toml config file. If not \
                 found reverts to the input file path",
                "[Path for the configuration file]");

    opts
}

fn execute(opts: &Options) -> FmtResult<Summary> {
    let matches = try!(opts.parse(env::args().skip(1)));

    match try!(determine_operation(&matches)) {
        Operation::Help => {
            print_usage(&opts, "");
            Ok(Summary::new())
        }
        Operation::Version => {
            print_version();
            Ok(Summary::new())
        }
        Operation::ConfigHelp => {
            Config::print_docs();
            Ok(Summary::new())
        }
        Operation::Stdin { input, config_path } => {
            // try to read config from local directory
            let (mut config, _) = match_cli_path_or_file(config_path, &env::current_dir().unwrap())
                .expect("Error resolving config");

            // write_mode is always Plain for Stdin.
            config.write_mode = WriteMode::Plain;

            Ok(run(Input::Text(input), &config))
        }
        Operation::Format { files, config_path } => {
            let options = try!(CliOptions::from_matches(&matches));
            let mut config = Config::default();
            let mut path = None;
            // Load the config path file if provided
            if let Some(config_file) = config_path {
                let (cfg_tmp, path_tmp) = resolve_config(config_file.as_ref())
                    .expect(&format!("Error resolving config for {:?}", config_file));
                config = cfg_tmp;
                path = path_tmp;
            };
            if let Some(path) = path.as_ref() {
                println!("Using rustfmt config file {}", path.display());
            }

            let mut error_summary = Summary::new();
            for file in files {
                // Check the file directory if the config-path could not be read or not provided
                if path.is_none() {
                    let (config_tmp, path_tmp) = resolve_config(file.parent().unwrap())
                        .expect(&format!("Error resolving config for {}", file.display()));
                    if let Some(path) = path_tmp.as_ref() {
                        println!("Using rustfmt config file {} for {}",
                                 path.display(),
                                 file.display());
                    }
                    config = config_tmp;
                }

                options.apply_to(&mut config);
                error_summary.add(run(Input::File(file), &config));
            }
            Ok(error_summary)
        }
    }
}

fn main() {
    let _ = env_logger::init();

    let opts = make_opts();

    let exit_code = match execute(&opts) {
        Ok(summary) => {
            if summary.has_operational_errors() {
                1
            } else if summary.has_parsing_errors() {
                2
            } else if summary.has_formatting_errors() {
                3
            } else {
                assert!(summary.has_no_errors());
                0
            }
        }
        Err(e) => {
            print_usage(&opts, &e.to_string());
            1
        }
    };
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
                         env::args_os().next().unwrap().to_string_lossy());
    println!("{}", opts.usage(&reason));
}

fn print_version() {
    println!("{}.{}.{}{}",
             option_env!("CARGO_PKG_VERSION_MAJOR").unwrap_or("X"),
             option_env!("CARGO_PKG_VERSION_MINOR").unwrap_or("X"),
             option_env!("CARGO_PKG_VERSION_PATCH").unwrap_or("X"),
             option_env!("CARGO_PKG_VERSION_PRE").unwrap_or(""));
}

fn determine_operation(matches: &Matches) -> FmtResult<Operation> {
    if matches.opt_present("h") {
        return Ok(Operation::Help);
    }

    if matches.opt_present("config-help") {
        return Ok(Operation::ConfigHelp);
    }

    if matches.opt_present("version") {
        return Ok(Operation::Version);
    }

    // Read the config_path and convert to parent dir if a file is provided.
    let config_path: Option<PathBuf> = matches.opt_str("config-path")
        .map(PathBuf::from)
        .and_then(|dir| {
            if dir.is_file() {
                return dir.parent().map(|v| v.into());
            }
            Some(dir)
        });

    // if no file argument is supplied, read from stdin
    if matches.free.is_empty() {

        let mut buffer = String::new();
        try!(io::stdin().read_to_string(&mut buffer));

        return Ok(Operation::Stdin {
            input: buffer,
            config_path: config_path,
        });
    }

    let files: Vec<_> = matches.free.iter().map(PathBuf::from).collect();

    Ok(Operation::Format {
        files: files,
        config_path: config_path,
    })
}
