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

extern crate env_logger;
extern crate failure;
extern crate getopts;
extern crate rustfmt_nightly as rustfmt;

use std::env;
use std::fs::File;
use std::io::{self, stdout, Read, Write};
use std::path::PathBuf;

use failure::err_msg;

use getopts::{Matches, Options};

use rustfmt::{
    emit_post_matter, emit_pre_matter, load_config, CliOptions, Config, FmtResult, WriteMode,
    WRITE_MODE_LIST,
};
use rustfmt::{format_and_emit_report, FileName, Input, Summary};

fn main() {
    env_logger::init();
    let opts = make_opts();

    let exit_code = match execute(&opts) {
        Ok((write_mode, summary)) => {
            if summary.has_operational_errors() || summary.has_parsing_errors()
                || (summary.has_diff && write_mode == WriteMode::Check)
            {
                1
            } else {
                0
            }
        }
        Err(e) => {
            eprintln!("{}", e.to_string());
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

/// Rustfmt operations.
enum Operation {
    /// Format files and their child modules.
    Format {
        files: Vec<PathBuf>,
        minimal_config_path: Option<String>,
    },
    /// Print the help message.
    Help,
    // Print version information
    Version,
    /// Print detailed configuration help.
    ConfigHelp,
    /// Output default config to a file, or stdout if None
    ConfigOutputDefault {
        path: Option<String>,
    },
    /// No file specified, read from stdin
    Stdin {
        input: String,
    },
}

fn make_opts() -> Options {
    let mut opts = Options::new();

    // Sorted in alphabetical order.
    opts.optopt(
        "",
        "color",
        "Use colored output (if supported)",
        "[always|never|auto]",
    );
    opts.optflag(
        "",
        "config-help",
        "Show details of rustfmt configuration options",
    );
    opts.optopt(
        "",
        "config-path",
        "Recursively searches the given path for the rustfmt.toml config file. If not \
         found reverts to the input file path",
        "[Path for the configuration file]",
    );
    opts.opt(
        "",
        "dump-default-config",
        "Dumps default configuration to PATH. PATH defaults to stdout, if omitted.",
        "PATH",
        getopts::HasArg::Maybe,
        getopts::Occur::Optional,
    );
    opts.optopt(
        "",
        "dump-minimal-config",
        "Dumps configuration options that were checked during formatting to a file.",
        "PATH",
    );
    opts.optflag(
        "",
        "error-on-unformatted",
        "Error if unable to get comments or string literals within max_width, \
         or they are left with trailing whitespaces",
    );
    opts.optopt(
        "",
        "file-lines",
        "Format specified line ranges. See README for more detail on the JSON format.",
        "JSON",
    );
    opts.optflag(
        "",
        "verbose-diff",
        "Emit a more verbose diff, indicating the end of lines.",
    );
    opts.optflag("h", "help", "Show this message");
    opts.optflag("", "skip-children", "Don't reformat child modules");
    opts.optflag(
        "",
        "unstable-features",
        "Enables unstable features. Only available on nightly channel",
    );
    opts.optflag("v", "verbose", "Print verbose output");
    opts.optflag("V", "version", "Show version information");
    opts.optopt(
        "",
        "write-mode",
        "How to write output (not usable when piping from stdin)",
        WRITE_MODE_LIST,
    );

    opts
}

fn execute(opts: &Options) -> FmtResult<(WriteMode, Summary)> {
    let matches = opts.parse(env::args().skip(1))?;

    match determine_operation(&matches)? {
        Operation::Help => {
            print_usage_to_stdout(opts, "");
            Summary::print_exit_codes();
            Ok((WriteMode::None, Summary::default()))
        }
        Operation::Version => {
            print_version();
            Ok((WriteMode::None, Summary::default()))
        }
        Operation::ConfigHelp => {
            Config::print_docs(&mut stdout(), matches.opt_present("unstable-features"));
            Ok((WriteMode::None, Summary::default()))
        }
        Operation::ConfigOutputDefault { path } => {
            let toml = Config::default().all_options().to_toml().map_err(err_msg)?;
            if let Some(path) = path {
                let mut file = File::create(path)?;
                file.write_all(toml.as_bytes())?;
            } else {
                io::stdout().write_all(toml.as_bytes())?;
            }
            Ok((WriteMode::None, Summary::default()))
        }
        Operation::Stdin { input } => {
            // try to read config from local directory
            let options = CliOptions::from_matches(&matches)?;
            let (mut config, _) = load_config(None, Some(&options))?;

            // write_mode is always Plain for Stdin.
            config.set().write_mode(WriteMode::Plain);

            // parse file_lines
            if let Some(ref file_lines) = matches.opt_str("file-lines") {
                config
                    .set()
                    .file_lines(file_lines.parse().map_err(err_msg)?);
                for f in config.file_lines().files() {
                    match *f {
                        FileName::Custom(ref f) if f == "stdin" => {}
                        _ => eprintln!("Warning: Extra file listed in file_lines option '{}'", f),
                    }
                }
            }

            let mut error_summary = Summary::default();
            emit_pre_matter(&config)?;
            match format_and_emit_report(Input::Text(input), &config) {
                Ok(summary) => error_summary.add(summary),
                Err(_) => error_summary.add_operational_error(),
            }
            emit_post_matter(&config)?;

            Ok((WriteMode::Plain, error_summary))
        }
        Operation::Format {
            files,
            minimal_config_path,
        } => {
            let options = CliOptions::from_matches(&matches)?;
            format(files, minimal_config_path, options)
        }
    }
}

fn format(
    files: Vec<PathBuf>,
    minimal_config_path: Option<String>,
    options: CliOptions,
) -> FmtResult<(WriteMode, Summary)> {
    options.verify_file_lines(&files);
    let (config, config_path) = load_config(None, Some(&options))?;

    if config.verbose() {
        if let Some(path) = config_path.as_ref() {
            println!("Using rustfmt config file {}", path.display());
        }
    }

    emit_pre_matter(&config)?;
    let mut error_summary = Summary::default();

    for file in files {
        if !file.exists() {
            eprintln!("Error: file `{}` does not exist", file.to_str().unwrap());
            error_summary.add_operational_error();
        } else if file.is_dir() {
            eprintln!("Error: `{}` is a directory", file.to_str().unwrap());
            error_summary.add_operational_error();
        } else {
            // Check the file directory if the config-path could not be read or not provided
            let local_config = if config_path.is_none() {
                let (local_config, config_path) =
                    load_config(Some(file.parent().unwrap()), Some(&options))?;
                if local_config.verbose() {
                    if let Some(path) = config_path {
                        println!(
                            "Using rustfmt config file {} for {}",
                            path.display(),
                            file.display()
                        );
                    }
                }
                local_config
            } else {
                config.clone()
            };

            match format_and_emit_report(Input::File(file), &local_config) {
                Ok(summary) => error_summary.add(summary),
                Err(_) => {
                    error_summary.add_operational_error();
                    break;
                }
            }
        }
    }
    emit_post_matter(&config)?;

    // If we were given a path via dump-minimal-config, output any options
    // that were used during formatting as TOML.
    if let Some(path) = minimal_config_path {
        let mut file = File::create(path)?;
        let toml = config.used_options().to_toml().map_err(err_msg)?;
        file.write_all(toml.as_bytes())?;
    }

    Ok((config.write_mode(), error_summary))
}

fn print_usage_to_stdout(opts: &Options, reason: &str) {
    let sep = if reason.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", reason)
    };
    let msg = format!(
        "{}Format Rust code\n\nusage: {} [options] <file>...",
        sep,
        env::args_os().next().unwrap().to_string_lossy()
    );
    println!("{}", opts.usage(&msg));
}

fn print_version() {
    let version_info = format!(
        "{}-{}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
        include_str!(concat!(env!("OUT_DIR"), "/commit-info.txt"))
    );

    println!("rustfmt {}", version_info);
}

fn determine_operation(matches: &Matches) -> FmtResult<Operation> {
    if matches.opt_present("h") {
        return Ok(Operation::Help);
    }

    if matches.opt_present("config-help") {
        return Ok(Operation::ConfigHelp);
    }

    if matches.opt_present("dump-default-config") {
        // NOTE for some reason when configured with HasArg::Maybe + Occur::Optional opt_default
        // doesn't recognize `--foo bar` as a long flag with an argument but as a long flag with no
        // argument *plus* a free argument. Thus we check for that case in this branch -- this is
        // required for backward compatibility.
        if let Some(path) = matches.free.get(0) {
            return Ok(Operation::ConfigOutputDefault {
                path: Some(path.clone()),
            });
        } else {
            return Ok(Operation::ConfigOutputDefault {
                path: matches.opt_str("dump-default-config"),
            });
        }
    }

    if matches.opt_present("version") {
        return Ok(Operation::Version);
    }

    // If no path is given, we won't output a minimal config.
    let minimal_config_path = matches.opt_str("dump-minimal-config");

    // if no file argument is supplied, read from stdin
    if matches.free.is_empty() {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;

        return Ok(Operation::Stdin { input: buffer });
    }

    let files: Vec<_> = matches
        .free
        .iter()
        .map(|s| {
            let p = PathBuf::from(s);
            // we will do comparison later, so here tries to canonicalize first
            // to get the expected behavior.
            p.canonicalize().unwrap_or(p)
        })
        .collect();

    Ok(Operation::Format {
        files,
        minimal_config_path,
    })
}
