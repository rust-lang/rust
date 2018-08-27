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
#![feature(extern_prelude)]

extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate getopts;
extern crate rustfmt_nightly as rustfmt;

use std::env;
use std::fs::File;
use std::io::{self, stdout, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use failure::err_msg;

use getopts::{Matches, Options};

use rustfmt::{
    load_config, CliOptions, Color, Config, EmitMode, ErrorKind, FileLines, FileName, Input,
    Session, Verbosity,
};

fn main() {
    env_logger::init();
    let opts = make_opts();

    let exit_code = match execute(&opts) {
        Ok(code) => code,
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
    Help(HelpOp),
    // Print version information
    Version,
    /// Output default config to a file, or stdout if None
    ConfigOutputDefault {
        path: Option<String>,
    },
    /// No file specified, read from stdin
    Stdin {
        input: String,
    },
}

/// Arguments to `--help`
enum HelpOp {
    None,
    Config,
    FileLines,
}

fn make_opts() -> Options {
    let mut opts = Options::new();

    opts.optflag(
        "",
        "check",
        "Run in 'check' mode. Exits with 0 if input if formatted correctly. Exits \
         with 1 and prints a diff if formatting is required.",
    );
    let is_nightly = is_nightly();
    let emit_opts = if is_nightly {
        "[files|stdout|coverage|checkstyle]"
    } else {
        "[files|stdout]"
    };
    opts.optopt("", "emit", "What data to emit and how", emit_opts);
    opts.optflag("", "backup", "Backup any modified files.");
    opts.optopt(
        "",
        "config-path",
        "Recursively searches the given path for the rustfmt.toml config file. If not \
         found reverts to the input file path",
        "[Path for the configuration file]",
    );
    opts.optopt(
        "",
        "color",
        "Use colored output (if supported)",
        "[always|never|auto]",
    );
    opts.optopt(
        "",
        "print-config",
        "Dumps a default or minimal config to PATH. A minimal config is the \
         subset of the current config file used for formatting the current program.",
        "[minimal|default] PATH",
    );

    if is_nightly {
        opts.optflag(
            "",
            "unstable-features",
            "Enables unstable features. Only available on nightly channel.",
        );
        opts.optopt(
            "",
            "file-lines",
            "Format specified line ranges. Run with `--help=file-lines` for \
             more detail (unstable).",
            "JSON",
        );
        opts.optflag(
            "",
            "error-on-unformatted",
            "Error if unable to get comments or string literals within max_width, \
             or they are left with trailing whitespaces (unstable).",
        );
        opts.optflag(
            "",
            "skip-children",
            "Don't reformat child modules (unstable).",
        );
    }

    opts.optflag("v", "verbose", "Print verbose output");
    opts.optflag("q", "quiet", "Print less output");
    opts.optflag("V", "version", "Show version information");
    opts.optflagopt(
        "h",
        "help",
        "Show this message or help about a specific topic: `config` or `file-lines`",
        "=TOPIC",
    );

    opts
}

fn is_nightly() -> bool {
    option_env!("CFG_RELEASE_CHANNEL")
        .map(|c| c == "nightly" || c == "dev")
        .unwrap_or(false)
}

// Returned i32 is an exit code
fn execute(opts: &Options) -> Result<i32, failure::Error> {
    let matches = opts.parse(env::args().skip(1))?;
    let options = GetOptsOptions::from_matches(&matches)?;

    match determine_operation(&matches)? {
        Operation::Help(HelpOp::None) => {
            print_usage_to_stdout(opts, "");
            return Ok(0);
        }
        Operation::Help(HelpOp::Config) => {
            Config::print_docs(&mut stdout(), options.unstable_features);
            return Ok(0);
        }
        Operation::Help(HelpOp::FileLines) => {
            print_help_file_lines();
            return Ok(0);
        }
        Operation::Version => {
            print_version();
            return Ok(0);
        }
        Operation::ConfigOutputDefault { path } => {
            let toml = Config::default().all_options().to_toml().map_err(err_msg)?;
            if let Some(path) = path {
                let mut file = File::create(path)?;
                file.write_all(toml.as_bytes())?;
            } else {
                io::stdout().write_all(toml.as_bytes())?;
            }
            return Ok(0);
        }
        Operation::Stdin { input } => format_string(input, options),
        Operation::Format {
            files,
            minimal_config_path,
        } => format(files, minimal_config_path, options),
    }
}

fn format_string(input: String, options: GetOptsOptions) -> Result<i32, failure::Error> {
    // try to read config from local directory
    let (mut config, _) = load_config(Some(Path::new(".")), Some(options.clone()))?;

    // emit mode is always Stdout for Stdin.
    config.set().emit_mode(EmitMode::Stdout);
    config.set().verbose(Verbosity::Quiet);

    // parse file_lines
    config.set().file_lines(options.file_lines);
    for f in config.file_lines().files() {
        match *f {
            FileName::Stdin => {}
            _ => eprintln!("Warning: Extra file listed in file_lines option '{}'", f),
        }
    }

    let out = &mut stdout();
    let mut session = Session::new(config, Some(out));
    format_and_emit_report(&mut session, Input::Text(input));

    let exit_code = if session.has_operational_errors() || session.has_parsing_errors() {
        1
    } else {
        0
    };
    Ok(exit_code)
}

fn format(
    files: Vec<PathBuf>,
    minimal_config_path: Option<String>,
    options: GetOptsOptions,
) -> Result<i32, failure::Error> {
    options.verify_file_lines(&files);
    let (config, config_path) = load_config(None, Some(options.clone()))?;

    if config.verbose() == Verbosity::Verbose {
        if let Some(path) = config_path.as_ref() {
            println!("Using rustfmt config file {}", path.display());
        }
    }

    let out = &mut stdout();
    let mut session = Session::new(config, Some(out));

    for file in files {
        if !file.exists() {
            eprintln!("Error: file `{}` does not exist", file.to_str().unwrap());
            session.add_operational_error();
        } else if file.is_dir() {
            eprintln!("Error: `{}` is a directory", file.to_str().unwrap());
            session.add_operational_error();
        } else {
            // Check the file directory if the config-path could not be read or not provided
            if config_path.is_none() {
                let (local_config, config_path) =
                    load_config(Some(file.parent().unwrap()), Some(options.clone()))?;
                if local_config.verbose() == Verbosity::Verbose {
                    if let Some(path) = config_path {
                        println!(
                            "Using rustfmt config file {} for {}",
                            path.display(),
                            file.display()
                        );
                    }
                }

                session.override_config(local_config, |sess| {
                    format_and_emit_report(sess, Input::File(file))
                });
            } else {
                format_and_emit_report(&mut session, Input::File(file));
            }
        }
    }

    // If we were given a path via dump-minimal-config, output any options
    // that were used during formatting as TOML.
    if let Some(path) = minimal_config_path {
        let mut file = File::create(path)?;
        let toml = session.config.used_options().to_toml().map_err(err_msg)?;
        file.write_all(toml.as_bytes())?;
    }

    let exit_code = if session.has_operational_errors()
        || session.has_parsing_errors()
        || ((session.has_diff() || session.has_check_errors()) && options.check)
    {
        1
    } else {
        0
    };
    Ok(exit_code)
}

fn format_and_emit_report<T: Write>(session: &mut Session<T>, input: Input) {
    match session.format(input) {
        Ok(report) => {
            if report.has_warnings() {
                match term::stderr() {
                    Some(ref t)
                        if session.config.color().use_colored_tty()
                            && t.supports_color()
                            && t.supports_attr(term::Attr::Bold) =>
                    {
                        match report.fancy_print(term::stderr().unwrap()) {
                            Ok(..) => (),
                            Err(..) => panic!("Unable to write to stderr: {}", report),
                        }
                    }
                    _ => eprintln!("{}", report),
                }
            }
        }
        Err(msg) => {
            eprintln!("Error writing files: {}", msg);
            session.add_operational_error();
        }
    }
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

fn print_help_file_lines() {
    println!(
        "If you want to restrict reformatting to specific sets of lines, you can
use the `--file-lines` option. Its argument is a JSON array of objects
with `file` and `range` properties, where `file` is a file name, and
`range` is an array representing a range of lines like `[7,13]`. Ranges
are 1-based and inclusive of both end points. Specifying an empty array
will result in no files being formatted. For example,

```
rustfmt --file-lines '[
    {{\"file\":\"src/lib.rs\",\"range\":[7,13]}},
    {{\"file\":\"src/lib.rs\",\"range\":[21,29]}},
    {{\"file\":\"src/foo.rs\",\"range\":[10,11]}},
    {{\"file\":\"src/foo.rs\",\"range\":[15,15]}}]'
```

would format lines `7-13` and `21-29` of `src/lib.rs`, and lines `10-11`,
and `15` of `src/foo.rs`. No other files would be formatted, even if they
are included as out of line modules from `src/lib.rs`."
    );
}

fn print_version() {
    let version_info = format!(
        "{}-{}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
        include_str!(concat!(env!("OUT_DIR"), "/commit-info.txt"))
    );

    println!("rustfmt {}", version_info);
}

fn determine_operation(matches: &Matches) -> Result<Operation, ErrorKind> {
    if matches.opt_present("h") {
        let topic = matches.opt_str("h");
        if topic == None {
            return Ok(Operation::Help(HelpOp::None));
        } else if topic == Some("config".to_owned()) {
            return Ok(Operation::Help(HelpOp::Config));
        } else if topic == Some("file-lines".to_owned()) {
            return Ok(Operation::Help(HelpOp::FileLines));
        } else {
            println!("Unknown help topic: `{}`\n", topic.unwrap());
            return Ok(Operation::Help(HelpOp::None));
        }
    }

    let mut minimal_config_path = None;
    if let Some(ref kind) = matches.opt_str("print-config") {
        let path = matches.free.get(0).cloned();
        if kind == "default" {
            return Ok(Operation::ConfigOutputDefault { path });
        } else if kind == "minimal" {
            minimal_config_path = path;
            if minimal_config_path.is_none() {
                println!("WARNING: PATH required for `--print-config minimal`");
            }
        }
    }

    if matches.opt_present("version") {
        return Ok(Operation::Version);
    }

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
        }).collect();

    Ok(Operation::Format {
        files,
        minimal_config_path,
    })
}

const STABLE_EMIT_MODES: [EmitMode; 3] = [EmitMode::Files, EmitMode::Stdout, EmitMode::Diff];

/// Parsed command line options.
#[derive(Clone, Debug, Default)]
struct GetOptsOptions {
    skip_children: Option<bool>,
    quiet: bool,
    verbose: bool,
    config_path: Option<PathBuf>,
    emit_mode: EmitMode,
    backup: bool,
    check: bool,
    color: Option<Color>,
    file_lines: FileLines, // Default is all lines in all files.
    unstable_features: bool,
    error_on_unformatted: Option<bool>,
}

impl GetOptsOptions {
    pub fn from_matches(matches: &Matches) -> Result<GetOptsOptions, failure::Error> {
        let mut options = GetOptsOptions::default();
        options.verbose = matches.opt_present("verbose");
        options.quiet = matches.opt_present("quiet");
        if options.verbose && options.quiet {
            return Err(format_err!("Can't use both `--verbose` and `--quiet`"));
        }

        let rust_nightly = is_nightly();

        if rust_nightly {
            options.unstable_features = matches.opt_present("unstable-features");

            if options.unstable_features {
                if matches.opt_present("skip-children") {
                    options.skip_children = Some(true);
                }
                if matches.opt_present("error-on-unformatted") {
                    options.error_on_unformatted = Some(true);
                }
                if let Some(ref file_lines) = matches.opt_str("file-lines") {
                    options.file_lines = file_lines.parse().map_err(err_msg)?;
                }
            } else {
                let mut unstable_options = vec![];
                if matches.opt_present("skip-children") {
                    unstable_options.push("`--skip-children`");
                }
                if matches.opt_present("error-on-unformatted") {
                    unstable_options.push("`--error-on-unformatted`");
                }
                if matches.opt_present("file-lines") {
                    unstable_options.push("`--file-lines`");
                }
                if !unstable_options.is_empty() {
                    let s = if unstable_options.len() == 1 { "" } else { "s" };
                    return Err(format_err!(
                        "Unstable option{} ({}) used without `--unstable-features`",
                        s,
                        unstable_options.join(", "),
                    ));
                }
            }
        }

        options.config_path = matches.opt_str("config-path").map(PathBuf::from);

        options.check = matches.opt_present("check");
        if let Some(ref emit_str) = matches.opt_str("emit") {
            if options.check {
                return Err(format_err!("Invalid to use `--emit` and `--check`"));
            }
            if let Ok(emit_mode) = emit_mode_from_emit_str(emit_str) {
                options.emit_mode = emit_mode;
            } else {
                return Err(format_err!("Invalid value for `--emit`"));
            }
        }

        if matches.opt_present("backup") {
            options.backup = true;
        }

        if !rust_nightly {
            if !STABLE_EMIT_MODES.contains(&options.emit_mode) {
                return Err(format_err!(
                    "Invalid value for `--emit` - using an unstable \
                     value without `--unstable-features`",
                ));
            }
        }

        if let Some(ref color) = matches.opt_str("color") {
            match Color::from_str(color) {
                Ok(color) => options.color = Some(color),
                _ => return Err(format_err!("Invalid color: {}", color)),
            }
        }

        Ok(options)
    }

    fn verify_file_lines(&self, files: &[PathBuf]) {
        for f in self.file_lines.files() {
            match *f {
                FileName::Real(ref f) if files.contains(f) => {}
                FileName::Real(_) => {
                    eprintln!("Warning: Extra file listed in file_lines option '{}'", f)
                }
                FileName::Stdin => eprintln!("Warning: Not a file '{}'", f),
            }
        }
    }
}

impl CliOptions for GetOptsOptions {
    fn apply_to(self, config: &mut Config) {
        if self.verbose {
            config.set().verbose(Verbosity::Verbose);
        } else if self.quiet {
            config.set().verbose(Verbosity::Quiet);
        } else {
            config.set().verbose(Verbosity::Normal);
        }
        config.set().file_lines(self.file_lines);
        config.set().unstable_features(self.unstable_features);
        if let Some(skip_children) = self.skip_children {
            config.set().skip_children(skip_children);
        }
        if let Some(error_on_unformatted) = self.error_on_unformatted {
            config.set().error_on_unformatted(error_on_unformatted);
        }
        if self.check {
            config.set().emit_mode(EmitMode::Diff);
        } else {
            config.set().emit_mode(self.emit_mode);
        }
        if self.backup {
            config.set().make_backup(true);
        }
        if let Some(color) = self.color {
            config.set().color(color);
        }
    }

    fn config_path(&self) -> Option<&Path> {
        self.config_path.as_ref().map(|p| &**p)
    }
}

fn emit_mode_from_emit_str(emit_str: &str) -> Result<EmitMode, failure::Error> {
    match emit_str {
        "files" => Ok(EmitMode::Files),
        "stdout" => Ok(EmitMode::Stdout),
        "coverage" => Ok(EmitMode::Coverage),
        "checkstyle" => Ok(EmitMode::Checkstyle),
        _ => Err(format_err!("Invalid value for `--emit`")),
    }
}
