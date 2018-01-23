// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]
#![cfg(not(test))]

extern crate env_logger;
extern crate getopts;
extern crate rustfmt_nightly as rustfmt;

use std::{env, error};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use getopts::{Matches, Options};

use rustfmt::{run, FileName, Input, Summary};
use rustfmt::config::{get_toml_path, Color, Config, WriteMode};
use rustfmt::file_lines::FileLines;

type FmtError = Box<error::Error + Send + Sync>;
type FmtResult<T> = std::result::Result<T, FmtError>;

/// Rustfmt operations.
enum Operation {
    /// Format files and their child modules.
    Format {
        files: Vec<PathBuf>,
        config_path: Option<PathBuf>,
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
        config_path: Option<PathBuf>,
    },
}

/// Parsed command line options.
#[derive(Clone, Debug, Default)]
struct CliOptions {
    skip_children: bool,
    verbose: bool,
    write_mode: Option<WriteMode>,
    color: Option<Color>,
    file_lines: FileLines, // Default is all lines in all files.
    unstable_features: bool,
    error_on_unformatted: bool,
}

impl CliOptions {
    fn from_matches(matches: &Matches) -> FmtResult<CliOptions> {
        let mut options = CliOptions::default();
        options.skip_children = matches.opt_present("skip-children");
        options.verbose = matches.opt_present("verbose");
        let unstable_features = matches.opt_present("unstable-features");
        let rust_nightly = option_env!("CFG_RELEASE_CHANNEL")
            .map(|c| c == "nightly")
            .unwrap_or(false);
        if unstable_features && !rust_nightly {
            return Err(FmtError::from(
                "Unstable features are only available on Nightly channel",
            ));
        } else {
            options.unstable_features = unstable_features;
        }

        if let Some(ref write_mode) = matches.opt_str("write-mode") {
            if let Ok(write_mode) = WriteMode::from_str(write_mode) {
                options.write_mode = Some(write_mode);
            } else {
                return Err(FmtError::from(format!(
                    "Invalid write-mode: {}",
                    write_mode
                )));
            }
        }

        if let Some(ref color) = matches.opt_str("color") {
            match Color::from_str(color) {
                Ok(color) => options.color = Some(color),
                _ => return Err(FmtError::from(format!("Invalid color: {}", color))),
            }
        }

        if let Some(ref file_lines) = matches.opt_str("file-lines") {
            options.file_lines = file_lines.parse()?;
        }

        if matches.opt_present("error-on-unformatted") {
            options.error_on_unformatted = true;
        }

        Ok(options)
    }

    fn apply_to(self, config: &mut Config) {
        config.set().skip_children(self.skip_children);
        config.set().verbose(self.verbose);
        config.set().file_lines(self.file_lines);
        config.set().unstable_features(self.unstable_features);
        config.set().error_on_unformatted(self.error_on_unformatted);
        if let Some(write_mode) = self.write_mode {
            config.set().write_mode(write_mode);
        }
        if let Some(color) = self.color {
            config.set().color(color);
        }
    }
}

/// read the given config file path recursively if present else read the project file path
fn match_cli_path_or_file(
    config_path: Option<PathBuf>,
    input_file: &Path,
) -> FmtResult<(Config, Option<PathBuf>)> {
    if let Some(config_file) = config_path {
        let toml = Config::from_toml_path(config_file.as_ref())?;
        return Ok((toml, Some(config_file)));
    }
    Config::from_resolved_toml_path(input_file).map_err(FmtError::from)
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
    opts.optopt(
        "",
        "dump-default-config",
        "Dumps default configuration to PATH. PATH defaults to stdout, if omitted.",
        "PATH",
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
        "[replace|overwrite|display|plain|diff|coverage|checkstyle]",
    );

    opts
}

fn execute(opts: &Options) -> FmtResult<Summary> {
    let matches = opts.parse(env::args().skip(1))?;

    match determine_operation(&matches)? {
        Operation::Help => {
            print_usage_to_stdout(opts, "");
            Summary::print_exit_codes();
            Ok(Summary::default())
        }
        Operation::Version => {
            print_version();
            Ok(Summary::default())
        }
        Operation::ConfigHelp => {
            Config::print_docs();
            Ok(Summary::default())
        }
        Operation::ConfigOutputDefault { path } => {
            let toml = Config::default().all_options().to_toml()?;
            if let Some(path) = path {
                let mut file = File::create(path)?;
                file.write_all(toml.as_bytes())?;
            } else {
                io::stdout().write_all(toml.as_bytes())?;
            }
            Ok(Summary::default())
        }
        Operation::Stdin { input, config_path } => {
            // try to read config from local directory
            let (mut config, _) =
                match_cli_path_or_file(config_path, &env::current_dir().unwrap())?;

            // write_mode is always Plain for Stdin.
            config.set().write_mode(WriteMode::Plain);

            // parse file_lines
            if let Some(ref file_lines) = matches.opt_str("file-lines") {
                config.set().file_lines(file_lines.parse()?);
                for f in config.file_lines().files() {
                    match *f {
                        FileName::Custom(ref f) if f == "stdin" => {}
                        _ => eprintln!("Warning: Extra file listed in file_lines option '{}'", f),
                    }
                }
            }

            let mut error_summary = Summary::default();
            if config.version_meets_requirement(&mut error_summary) {
                error_summary.add(run(Input::Text(input), &config));
            }

            Ok(error_summary)
        }
        Operation::Format {
            files,
            config_path,
            minimal_config_path,
        } => {
            let options = CliOptions::from_matches(&matches)?;

            for f in options.file_lines.files() {
                match *f {
                    FileName::Real(ref f) if files.contains(f) => {}
                    FileName::Real(_) => {
                        eprintln!("Warning: Extra file listed in file_lines option '{}'", f)
                    }
                    _ => eprintln!("Warning: Not a file '{}'", f),
                }
            }

            let mut config = Config::default();
            // Load the config path file if provided
            if let Some(config_file) = config_path.as_ref() {
                config = Config::from_toml_path(config_file.as_ref())?;
            };

            if options.verbose {
                if let Some(path) = config_path.as_ref() {
                    println!("Using rustfmt config file {}", path.display());
                }
            }

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
                    if config_path.is_none() {
                        let (config_tmp, path_tmp) =
                            Config::from_resolved_toml_path(file.parent().unwrap())?;
                        if options.verbose {
                            if let Some(path) = path_tmp.as_ref() {
                                println!(
                                    "Using rustfmt config file {} for {}",
                                    path.display(),
                                    file.display()
                                );
                            }
                        }
                        config = config_tmp;
                    }

                    if !config.version_meets_requirement(&mut error_summary) {
                        break;
                    }

                    options.clone().apply_to(&mut config);
                    error_summary.add(run(Input::File(file), &config));
                }
            }

            // If we were given a path via dump-minimal-config, output any options
            // that were used during formatting as TOML.
            if let Some(path) = minimal_config_path {
                let mut file = File::create(path)?;
                let toml = config.used_options().to_toml()?;
                file.write_all(toml.as_bytes())?;
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
            } else if summary.has_diff {
                // should only happen in diff mode
                4
            } else {
                assert!(summary.has_no_errors());
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
    println!(
        "{}-nightly{}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
        include_str!(concat!(env!("OUT_DIR"), "/commit-info.txt"))
    )
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

    let config_path_not_found = |path: &str| -> FmtResult<Operation> {
        Err(FmtError::from(format!(
            "Error: unable to find a config file for the given path: `{}`",
            path
        )))
    };

    // Read the config_path and convert to parent dir if a file is provided.
    // If a config file cannot be found from the given path, return error.
    let config_path: Option<PathBuf> = match matches.opt_str("config-path").map(PathBuf::from) {
        Some(ref path) if !path.exists() => return config_path_not_found(path.to_str().unwrap()),
        Some(ref path) if path.is_dir() => {
            let config_file_path = get_toml_path(path)?;
            if config_file_path.is_some() {
                config_file_path
            } else {
                return config_path_not_found(path.to_str().unwrap());
            }
        }
        path => path,
    };

    // If no path is given, we won't output a minimal config.
    let minimal_config_path = matches.opt_str("dump-minimal-config");

    // if no file argument is supplied, read from stdin
    if matches.free.is_empty() {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;

        return Ok(Operation::Stdin {
            input: buffer,
            config_path: config_path,
        });
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
        files: files,
        config_path: config_path,
        minimal_config_path: minimal_config_path,
    })
}
