use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use colored::*;
use comments::ErrorMatch;
use crossbeam::queue::SegQueue;
use regex::Regex;

pub use crate::comments::Comments;

mod comments;

#[derive(Debug)]
pub struct Config {
    /// Arguments passed to the binary that is executed.
    pub args: Vec<String>,
    /// `None` to run on the host, otherwise a target triple
    pub target: Option<String>,
    /// Filters applied to stderr output before processing it
    pub stderr_filters: Filter,
    /// Filters applied to stdout output before processing it
    pub stdout_filters: Filter,
    /// The folder in which to start searching for .rs files
    pub root_dir: PathBuf,
    pub mode: Mode,
    pub program: PathBuf,
    pub output_conflict_handling: OutputConflictHandling,
}

#[derive(Debug)]
pub enum OutputConflictHandling {
    /// The default: emit a diff of the expected/actual output.
    Error,
    /// Ignore mismatches in the stderr/stdout files.
    Ignore,
    /// Instead of erroring if the stderr/stdout differs from the expected
    /// automatically replace it with the found output (after applying filters).
    Bless,
}

pub type Filter = Vec<(Regex, &'static str)>;

pub fn run_tests(config: Config) {
    eprintln!("   Compiler flags: {:?}", config.args);

    // Get the triple with which to run the tests
    let target = config.target.clone().unwrap_or_else(|| config.get_host());

    // A queue for files or folders to process
    let todo = SegQueue::new();
    todo.push(config.root_dir.clone());

    // Some statistics and failure reports.
    let failures = Mutex::new(vec![]);
    let succeeded = AtomicUsize::default();
    let ignored = AtomicUsize::default();

    crossbeam::scope(|s| {
        for _ in 0..std::thread::available_parallelism().unwrap().get() {
            s.spawn(|_| {
                while let Some(path) = todo.pop() {
                    // Collect everything inside directories
                    if path.is_dir() {
                        for entry in std::fs::read_dir(path).unwrap() {
                            todo.push(entry.unwrap().path());
                        }
                        continue;
                    }
                    // Only look at .rs files
                    if !path.extension().map(|ext| ext == "rs").unwrap_or(false) {
                        continue;
                    }
                    let comments = Comments::parse_file(&path);
                    // Ignore file if only/ignore rules do (not) apply
                    if ignore_file(&comments, &target) {
                        ignored.fetch_add(1, Ordering::Relaxed);
                        eprintln!("{} .. {}", path.display(), "ignored".yellow());
                        continue;
                    }
                    // Run the test for all revisions
                    for revision in
                        comments.revisions.clone().unwrap_or_else(|| vec![String::new()])
                    {
                        let (m, errors) = run_test(&path, &config, &target, &revision, &comments);

                        // Using `format` to prevent messages from threads from getting intermingled.
                        let mut msg = format!("{} ", path.display());
                        if !revision.is_empty() {
                            write!(msg, "(revision `{revision}`) ").unwrap();
                        }
                        write!(msg, "... ").unwrap();
                        if errors.is_empty() {
                            eprintln!("{msg}{}", "ok".green());
                            succeeded.fetch_add(1, Ordering::Relaxed);
                        } else {
                            eprintln!("{msg}{}", "FAILED".red().bold());
                            failures.lock().unwrap().push((path.clone(), m, revision, errors));
                        }
                    }
                }
            });
        }
    })
    .unwrap();

    // Print all errors in a single thread to show reliable output
    let failures = failures.into_inner().unwrap();
    let succeeded = succeeded.load(Ordering::Relaxed);
    let ignored = ignored.load(Ordering::Relaxed);
    if !failures.is_empty() {
        for (path, miri, revision, errors) in &failures {
            eprintln!();
            eprint!("{}", path.display().to_string().underline());
            if !revision.is_empty() {
                eprint!(" (revision `{}`)", revision);
            }
            eprint!(" {}", "FAILED".red());
            eprintln!();
            eprintln!("command: {:?}", miri);
            eprintln!();
            let mut dump_stderr = None;
            for error in errors {
                match error {
                    Error::ExitStatus(mode, exit_status) => eprintln!("{mode:?} got {exit_status}"),
                    Error::PatternNotFound { stderr, pattern, definition_line } => {
                        eprintln!("`{pattern}` {} in stderr output", "not found".red());
                        eprintln!(
                            "expected because of pattern here: {}:{definition_line}",
                            path.display()
                        );
                        dump_stderr = Some(stderr.clone())
                    }
                    Error::NoPatternsFound =>
                        eprintln!("{}", "no error patterns found in failure test".red()),
                    Error::PatternFoundInPassTest =>
                        eprintln!("{}", "error pattern found in success test".red()),
                    Error::OutputDiffers { path, actual, expected } => {
                        dump_stderr = None;
                        eprintln!("actual output differed from expected {}", path.display());
                        eprintln!("{}", pretty_assertions::StrComparison::new(expected, actual));
                        eprintln!()
                    }
                }
                eprintln!();
            }
            if let Some(stderr) = dump_stderr {
                eprintln!("actual stderr:");
                eprintln!("{}", stderr);
                eprintln!();
            }
        }
        eprintln!(
            "{} tests failed, {} tests passed, {} ignored",
            failures.len().to_string().red().bold(),
            succeeded.to_string().green(),
            ignored.to_string().yellow()
        );
        std::process::exit(1);
    }
    eprintln!();
    eprintln!(
        "test result: {}. {} tests passed, {} ignored",
        "ok".green(),
        succeeded.to_string().green(),
        ignored.to_string().yellow()
    );
    eprintln!();
}

#[derive(Debug)]
pub enum Error {
    /// Got an invalid exit status for the given mode.
    ExitStatus(Mode, ExitStatus),
    PatternNotFound {
        stderr: String,
        pattern: String,
        definition_line: usize,
    },
    /// A ui test checking for failure does not have any failure patterns
    NoPatternsFound,
    /// A ui test checking for success has failure patterns
    PatternFoundInPassTest,
    /// Stderr/Stdout differed from the `.stderr`/`.stdout` file present.
    OutputDiffers {
        path: PathBuf,
        actual: String,
        expected: String,
    },
}

pub type Errors = Vec<Error>;

fn run_test(
    path: &Path,
    config: &Config,
    target: &str,
    revision: &str,
    comments: &Comments,
) -> (Command, Errors) {
    // Run miri
    let mut miri = Command::new(&config.program);
    miri.args(config.args.iter());
    miri.arg(path);
    if !revision.is_empty() {
        miri.arg(format!("--cfg={revision}"));
    }
    for arg in &comments.compile_flags {
        miri.arg(arg);
    }
    for (k, v) in &comments.env_vars {
        miri.env(k, v);
    }
    let output = miri.output().expect("could not execute miri");
    let mut errors = config.mode.ok(output.status);
    // Check output files (if any)
    let revised = |extension: &str| {
        if revision.is_empty() {
            extension.to_string()
        } else {
            format!("{}.{}", revision, extension)
        }
    };
    // Check output files against actual output
    check_output(
        &output.stderr,
        path,
        &mut errors,
        revised("stderr"),
        target,
        &config.stderr_filters,
        &config,
        comments,
    );
    check_output(
        &output.stdout,
        path,
        &mut errors,
        revised("stdout"),
        target,
        &config.stdout_filters,
        &config,
        comments,
    );
    // Check error annotations in the source against output
    check_annotations(&output.stderr, &mut errors, config, revision, comments);
    (miri, errors)
}

pub fn check_annotations(
    unnormalized_stderr: &[u8],
    errors: &mut Errors,
    config: &Config,
    revision: &str,
    comments: &Comments,
) {
    let unnormalized_stderr = std::str::from_utf8(unnormalized_stderr).unwrap();
    let mut found_annotation = false;
    if let Some((ref error_pattern, definition_line)) = comments.error_pattern {
        if !unnormalized_stderr.contains(error_pattern) {
            errors.push(Error::PatternNotFound {
                stderr: unnormalized_stderr.to_string(),
                pattern: error_pattern.to_string(),
                definition_line,
            });
        }
        found_annotation = true;
    }
    for &ErrorMatch { ref matched, revision: ref rev, definition_line } in &comments.error_matches {
        // FIXME: check that the error happens on the marked line

        if let Some(rev) = rev {
            if rev != revision {
                continue;
            }
        }

        if !unnormalized_stderr.contains(matched) {
            errors.push(Error::PatternNotFound {
                stderr: unnormalized_stderr.to_string(),
                pattern: matched.to_string(),
                definition_line,
            });
        }
        found_annotation = true;
    }
    match (config.mode, found_annotation) {
        (Mode::Pass, true) | (Mode::Panic, true) => errors.push(Error::PatternFoundInPassTest),
        (Mode::Fail, false) => errors.push(Error::NoPatternsFound),
        _ => {}
    };
}

fn check_output(
    output: &[u8],
    path: &Path,
    errors: &mut Errors,
    kind: String,
    target: &str,
    filters: &Filter,
    config: &Config,
    comments: &Comments,
) {
    let output = std::str::from_utf8(&output).unwrap();
    let output = normalize(path, output, filters, comments);
    let path = output_path(path, comments, kind, target);
    match config.output_conflict_handling {
        OutputConflictHandling::Bless =>
            if output.is_empty() {
                let _ = std::fs::remove_file(path);
            } else {
                std::fs::write(path, &output).unwrap();
            },
        OutputConflictHandling::Error => {
            let expected_output = std::fs::read_to_string(&path).unwrap_or_default();
            if output != expected_output {
                errors.push(Error::OutputDiffers {
                    path,
                    actual: output,
                    expected: expected_output,
                });
            }
        }
        OutputConflictHandling::Ignore => {}
    }
}

fn output_path(path: &Path, comments: &Comments, kind: String, target: &str) -> PathBuf {
    if comments.stderr_per_bitwidth {
        return path.with_extension(format!("{}.{kind}", get_pointer_width(target)));
    }
    path.with_extension(kind)
}

fn ignore_file(comments: &Comments, target: &str) -> bool {
    for s in &comments.ignore {
        if target.contains(s) {
            return true;
        }
        if get_pointer_width(target) == s {
            return true;
        }
    }
    for s in &comments.only {
        if !target.contains(s) {
            return true;
        }
        if get_pointer_width(target) != s {
            return true;
        }
    }
    false
}

// Taken 1:1 from compiletest-rs
fn get_pointer_width(triple: &str) -> &'static str {
    if (triple.contains("64") && !triple.ends_with("gnux32") && !triple.ends_with("gnu_ilp32"))
        || triple.starts_with("s390x")
    {
        "64bit"
    } else if triple.starts_with("avr") {
        "16bit"
    } else {
        "32bit"
    }
}

fn normalize(path: &Path, text: &str, filters: &Filter, comments: &Comments) -> String {
    // Useless paths
    let mut text = text.replace(&path.parent().unwrap().display().to_string(), "$DIR");
    if let Some(lib_path) = option_env!("RUSTC_LIB_PATH") {
        text = text.replace(lib_path, "RUSTLIB");
    }

    for (regex, replacement) in filters.iter() {
        text = regex.replace_all(&text, *replacement).to_string();
    }

    for (from, to) in &comments.normalize_stderr {
        text = from.replace_all(&text, to).to_string();
    }
    text
}

impl Config {
    fn get_host(&self) -> String {
        rustc_version::VersionMeta::for_command(std::process::Command::new(&self.program))
            .expect("failed to parse rustc version info")
            .host
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Mode {
    // The test passes a full execution of the rustc driver
    Pass,
    // The rustc driver panicked
    Panic,
    // The rustc driver emitted an error
    Fail,
}

impl Mode {
    fn ok(self, status: ExitStatus) -> Errors {
        match (status.code().unwrap(), self) {
            (1, Mode::Fail) | (101, Mode::Panic) | (0, Mode::Pass) => vec![],
            _ => vec![Error::ExitStatus(self, status)],
        }
    }
}
