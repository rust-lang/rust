#![feature(rustc_private)]

use anyhow::{format_err, Result};

use io::Error as IoError;
use thiserror::Error;

use rustfmt_nightly as rustfmt;

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, stdout, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use getopts::{Matches, Options};

use crate::rustfmt::{
    load_config, CliOptions, Color, Config, Edition, EmitMode, FileLines, FileName,
    FormatReportFormatterBuilder, Input, Session, Verbosity,
};

const BUG_REPORT_URL: &str = "https://github.com/rust-lang/rustfmt/issues/new?labels=bug";

// N.B. these crates are loaded from the sysroot, so they need extern crate.
extern crate rustc_driver;

fn main() {
    rustc_driver::install_ice_hook(BUG_REPORT_URL, |_| ());

    env_logger::Builder::from_env("RUSTFMT_LOG").init();
    let opts = make_opts();

    let exit_code = match execute(&opts) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{:#}", e);
            1
        }
    };
    // Make sure standard output is flushed before we exit.
    std::io::stdout().flush().unwrap();

    // Exit with given exit code.
    //
    // NOTE: this immediately terminates the process without doing any cleanup,
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
    /// Print version information
    Version,
    /// Output default config to a file, or stdout if None
    ConfigOutputDefault { path: Option<String> },
    /// Output current config (as if formatting to a file) to stdout
    ConfigOutputCurrent { path: Option<String> },
    /// No file specified, read from stdin
    Stdin { input: String },
}

/// Rustfmt operations errors.
#[derive(Error, Debug)]
pub enum OperationError {
    /// An unknown help topic was requested.
    #[error("Unknown help topic: `{0}`.")]
    UnknownHelpTopic(String),
    /// An unknown print-config option was requested.
    #[error("Unknown print-config option: `{0}`.")]
    UnknownPrintConfigTopic(String),
    /// Attempt to generate a minimal config from standard input.
    #[error("The `--print-config=minimal` option doesn't work with standard input.")]
    MinimalPathWithStdin,
    /// An io error during reading or writing.
    #[error("{0}")]
    IoError(IoError),
    /// Attempt to use --emit with a mode which is not currently
    /// supported with stdandard input.
    #[error("Emit mode {0} not supported with standard output.")]
    StdinBadEmit(EmitMode),
}

impl From<IoError> for OperationError {
    fn from(e: IoError) -> OperationError {
        OperationError::IoError(e)
    }
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
        "Run in 'check' mode. Exits with 0 if input is formatted correctly. Exits \
         with 1 and prints a diff if formatting is required.",
    );
    let is_nightly = is_nightly();
    let emit_opts = if is_nightly {
        "[files|stdout|coverage|checkstyle|json]"
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
    opts.optopt("", "edition", "Rust edition to use", "[2015|2018|2021]");
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
         subset of the current config file used for formatting the current program. \
         `current` writes to stdout current config as if formatting the file at PATH.",
        "[default|minimal|current] PATH",
    );
    opts.optflag(
        "l",
        "files-with-diff",
        "Prints the names of mismatched files that were formatted. Prints the names of \
         files that would be formatted when used with `--check` mode. ",
    );
    opts.optmulti(
        "",
        "config",
        "Set options from command line. These settings take priority over .rustfmt.toml",
        "[key1=val1,key2=val2...]",
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
    let help_topics = if is_nightly {
        "`config` or `file-lines`"
    } else {
        "`config`"
    };
    let mut help_topic_msg = "Show this message or help about a specific topic: ".to_owned();
    help_topic_msg.push_str(help_topics);

    opts.optflagopt("h", "help", &help_topic_msg, "=TOPIC");

    opts
}

fn is_nightly() -> bool {
    option_env!("CFG_RELEASE_CHANNEL").map_or(true, |c| c == "nightly" || c == "dev")
}

// Returned i32 is an exit code
fn execute(opts: &Options) -> Result<i32> {
    let matches = opts.parse(env::args().skip(1))?;
    let options = GetOptsOptions::from_matches(&matches)?;

    match determine_operation(&matches)? {
        Operation::Help(HelpOp::None) => {
            print_usage_to_stdout(opts, "");
            Ok(0)
        }
        Operation::Help(HelpOp::Config) => {
            Config::print_docs(&mut stdout(), options.unstable_features);
            Ok(0)
        }
        Operation::Help(HelpOp::FileLines) => {
            print_help_file_lines();
            Ok(0)
        }
        Operation::Version => {
            print_version();
            Ok(0)
        }
        Operation::ConfigOutputDefault { path } => {
            let toml = Config::default().all_options().to_toml()?;
            if let Some(path) = path {
                let mut file = File::create(path)?;
                file.write_all(toml.as_bytes())?;
            } else {
                io::stdout().write_all(toml.as_bytes())?;
            }
            Ok(0)
        }
        Operation::ConfigOutputCurrent { path } => {
            let path = match path {
                Some(path) => path,
                None => return Err(format_err!("PATH required for `--print-config current`")),
            };

            let file = PathBuf::from(path);
            let file = file.canonicalize().unwrap_or(file);

            let (config, _) = load_config(Some(file.parent().unwrap()), Some(options))?;
            let toml = config.all_options().to_toml()?;
            io::stdout().write_all(toml.as_bytes())?;

            Ok(0)
        }
        Operation::Stdin { input } => format_string(input, options),
        Operation::Format {
            files,
            minimal_config_path,
        } => format(files, minimal_config_path, &options),
    }
}

fn format_string(input: String, options: GetOptsOptions) -> Result<i32> {
    // try to read config from local directory
    let (mut config, _) = load_config(Some(Path::new(".")), Some(options.clone()))?;

    if options.check {
        config.set().emit_mode(EmitMode::Diff);
    } else {
        match options.emit_mode {
            // Emit modes which work with standard input
            // None means default, which is Stdout.
            None | Some(EmitMode::Stdout) | Some(EmitMode::Checkstyle) | Some(EmitMode::Json) => {}
            Some(emit_mode) => {
                return Err(OperationError::StdinBadEmit(emit_mode).into());
            }
        }
        config
            .set()
            .emit_mode(options.emit_mode.unwrap_or(EmitMode::Stdout));
    }
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
    options: &GetOptsOptions,
) -> Result<i32> {
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
        let toml = session.config.used_options().to_toml()?;
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

fn format_and_emit_report<T: Write>(session: &mut Session<'_, T>, input: Input) {
    match session.format(input) {
        Ok(report) => {
            if report.has_warnings() {
                eprintln!(
                    "{}",
                    FormatReportFormatterBuilder::new(&report)
                        .enable_colors(should_print_with_colors(session))
                        .build()
                );
            }
        }
        Err(msg) => {
            eprintln!("Error writing files: {}", msg);
            session.add_operational_error();
        }
    }
}

fn should_print_with_colors<T: Write>(session: &mut Session<'_, T>) -> bool {
    match term::stderr() {
        Some(ref t)
            if session.config.color().use_colored_tty()
                && t.supports_color()
                && t.supports_attr(term::Attr::Bold) =>
        {
            true
        }
        _ => false,
    }
}

fn print_usage_to_stdout(opts: &Options, reason: &str) {
    let sep = if reason.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", reason)
    };
    let msg = format!(
        "{}Format Rust code\n\nusage: rustfmt [options] <file>...",
        sep
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

fn determine_operation(matches: &Matches) -> Result<Operation, OperationError> {
    if matches.opt_present("h") {
        let topic = matches.opt_str("h");
        if topic == None {
            return Ok(Operation::Help(HelpOp::None));
        } else if topic == Some("config".to_owned()) {
            return Ok(Operation::Help(HelpOp::Config));
        } else if topic == Some("file-lines".to_owned()) && is_nightly() {
            return Ok(Operation::Help(HelpOp::FileLines));
        } else {
            return Err(OperationError::UnknownHelpTopic(topic.unwrap()));
        }
    }
    let mut free_matches = matches.free.iter();

    let mut minimal_config_path = None;
    if let Some(kind) = matches.opt_str("print-config") {
        let path = free_matches.next().cloned();
        match kind.as_str() {
            "default" => return Ok(Operation::ConfigOutputDefault { path }),
            "current" => return Ok(Operation::ConfigOutputCurrent { path }),
            "minimal" => {
                minimal_config_path = path;
                if minimal_config_path.is_none() {
                    eprintln!("WARNING: PATH required for `--print-config minimal`.");
                }
            }
            _ => {
                return Err(OperationError::UnknownPrintConfigTopic(kind));
            }
        }
    }

    if matches.opt_present("version") {
        return Ok(Operation::Version);
    }

    let files: Vec<_> = free_matches
        .map(|s| {
            let p = PathBuf::from(s);
            // we will do comparison later, so here tries to canonicalize first
            // to get the expected behavior.
            p.canonicalize().unwrap_or(p)
        })
        .collect();

    // if no file argument is supplied, read from stdin
    if files.is_empty() {
        if minimal_config_path.is_some() {
            return Err(OperationError::MinimalPathWithStdin);
        }
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;

        return Ok(Operation::Stdin { input: buffer });
    }

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
    inline_config: HashMap<String, String>,
    emit_mode: Option<EmitMode>,
    backup: bool,
    check: bool,
    edition: Option<Edition>,
    color: Option<Color>,
    file_lines: FileLines, // Default is all lines in all files.
    unstable_features: bool,
    error_on_unformatted: Option<bool>,
    print_misformatted_file_names: bool,
}

impl GetOptsOptions {
    pub fn from_matches(matches: &Matches) -> Result<GetOptsOptions> {
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
                    options.file_lines = file_lines.parse()?;
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

        options.inline_config = matches
            .opt_strs("config")
            .iter()
            .flat_map(|config| config.split(','))
            .map(
                |key_val| match key_val.char_indices().find(|(_, ch)| *ch == '=') {
                    Some((middle, _)) => {
                        let (key, val) = (&key_val[..middle], &key_val[middle + 1..]);
                        if !Config::is_valid_key_val(key, val) {
                            Err(format_err!("invalid key=val pair: `{}`", key_val))
                        } else {
                            Ok((key.to_string(), val.to_string()))
                        }
                    }

                    None => Err(format_err!(
                        "--config expects comma-separated list of key=val pairs, found `{}`",
                        key_val
                    )),
                },
            )
            .collect::<Result<HashMap<_, _>, _>>()?;

        options.check = matches.opt_present("check");
        if let Some(ref emit_str) = matches.opt_str("emit") {
            if options.check {
                return Err(format_err!("Invalid to use `--emit` and `--check`"));
            }

            options.emit_mode = Some(emit_mode_from_emit_str(emit_str)?);
        }

        if let Some(ref edition_str) = matches.opt_str("edition") {
            options.edition = Some(edition_from_edition_str(edition_str)?);
        }

        if matches.opt_present("backup") {
            options.backup = true;
        }

        if matches.opt_present("files-with-diff") {
            options.print_misformatted_file_names = true;
        }

        if !rust_nightly {
            if let Some(ref emit_mode) = options.emit_mode {
                if !STABLE_EMIT_MODES.contains(emit_mode) {
                    return Err(format_err!(
                        "Invalid value for `--emit` - using an unstable \
                         value without `--unstable-features`",
                    ));
                }
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
        if let Some(edition) = self.edition {
            config.set().edition(edition);
        }
        if self.check {
            config.set().emit_mode(EmitMode::Diff);
        } else if let Some(emit_mode) = self.emit_mode {
            config.set().emit_mode(emit_mode);
        }
        if self.backup {
            config.set().make_backup(true);
        }
        if let Some(color) = self.color {
            config.set().color(color);
        }
        if self.print_misformatted_file_names {
            config.set().print_misformatted_file_names(true);
        }

        for (key, val) in self.inline_config {
            config.override_value(&key, &val);
        }
    }

    fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }
}

fn edition_from_edition_str(edition_str: &str) -> Result<Edition> {
    match edition_str {
        "2015" => Ok(Edition::Edition2015),
        "2018" => Ok(Edition::Edition2018),
        "2021" => Ok(Edition::Edition2021),
        "2024" => Ok(Edition::Edition2024),
        _ => Err(format_err!("Invalid value for `--edition`")),
    }
}

fn emit_mode_from_emit_str(emit_str: &str) -> Result<EmitMode> {
    match emit_str {
        "files" => Ok(EmitMode::Files),
        "stdout" => Ok(EmitMode::Stdout),
        "coverage" => Ok(EmitMode::Coverage),
        "checkstyle" => Ok(EmitMode::Checkstyle),
        "json" => Ok(EmitMode::Json),
        _ => Err(format_err!("Invalid value for `--emit`")),
    }
}
