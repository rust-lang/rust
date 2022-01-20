use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::iter::Peekable;
use std::mem;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::Chars;
use std::thread;

use crate::config::{Color, Config, EmitMode, FileName, NewlineStyle, ReportTactic};
use crate::formatting::{ReportedErrors, SourceFile};
use crate::rustfmt_diff::{make_diff, print_diff, DiffLine, Mismatch, ModifiedChunk, OutputWriter};
use crate::source_file;
use crate::{is_nightly_channel, FormatReport, FormatReportFormatterBuilder, Input, Session};

use rustfmt_config_proc_macro::nightly_only_test;

mod configuration_snippet;
mod mod_resolver;
mod parser;

const DIFF_CONTEXT_SIZE: usize = 3;

// A list of files on which we want to skip testing.
const SKIP_FILE_WHITE_LIST: &[&str] = &[
    // We want to make sure that the `skip_children` is correctly working,
    // so we do not want to test this file directly.
    "configs/skip_children/foo/mod.rs",
    "issue-3434/no_entry.rs",
    "issue-3665/sub_mod.rs",
    // Testing for issue-3779
    "issue-3779/ice.rs",
    // These files and directory are a part of modules defined inside `cfg_if!`.
    "cfg_if/mod.rs",
    "cfg_if/detect",
    "issue-3253/foo.rs",
    "issue-3253/bar.rs",
    "issue-3253/paths",
    // These files and directory are a part of modules defined inside `cfg_attr(..)`.
    "cfg_mod/dir",
    "cfg_mod/bar.rs",
    "cfg_mod/foo.rs",
    "cfg_mod/wasm32.rs",
    "skip/foo.rs",
];

fn init_log() {
    let _ = env_logger::builder().is_test(true).try_init();
}

struct TestSetting {
    /// The size of the stack of the thread that run tests.
    stack_size: usize,
}

impl Default for TestSetting {
    fn default() -> Self {
        TestSetting {
            stack_size: 8_388_608, // 8MB
        }
    }
}

fn run_test_with<F>(test_setting: &TestSetting, f: F)
where
    F: FnOnce(),
    F: Send + 'static,
{
    thread::Builder::new()
        .stack_size(test_setting.stack_size)
        .spawn(f)
        .expect("Failed to create a test thread")
        .join()
        .expect("Failed to join a test thread")
}

fn is_subpath<P>(path: &Path, subpath: &P) -> bool
where
    P: AsRef<Path>,
{
    (0..path.components().count())
        .map(|i| {
            path.components()
                .skip(i)
                .take(subpath.as_ref().components().count())
        })
        .any(|c| c.zip(subpath.as_ref().components()).all(|(a, b)| a == b))
}

fn is_file_skip(path: &Path) -> bool {
    SKIP_FILE_WHITE_LIST
        .iter()
        .any(|file_path| is_subpath(path, file_path))
}

// Returns a `Vec` containing `PathBuf`s of files with an  `rs` extension in the
// given path. The `recursive` argument controls if files from subdirectories
// are also returned.
fn get_test_files(path: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut files = vec![];
    if path.is_dir() {
        for entry in fs::read_dir(path).expect(&format!(
            "couldn't read directory {}",
            path.to_str().unwrap()
        )) {
            let entry = entry.expect("couldn't get `DirEntry`");
            let path = entry.path();
            if path.is_dir() && recursive {
                files.append(&mut get_test_files(&path, recursive));
            } else if path.extension().map_or(false, |f| f == "rs") && !is_file_skip(&path) {
                files.push(path);
            }
        }
    }
    files
}

fn verify_config_used(path: &Path, config_name: &str) {
    for entry in fs::read_dir(path).expect(&format!(
        "couldn't read {} directory",
        path.to_str().unwrap()
    )) {
        let entry = entry.expect("couldn't get directory entry");
        let path = entry.path();
        if path.extension().map_or(false, |f| f == "rs") {
            // check if "// rustfmt-<config_name>:" appears in the file.
            let filebuf = BufReader::new(
                fs::File::open(&path)
                    .unwrap_or_else(|_| panic!("couldn't read file {}", path.display())),
            );
            assert!(
                filebuf
                    .lines()
                    .map(Result::unwrap)
                    .take_while(|l| l.starts_with("//"))
                    .any(|l| l.starts_with(&format!("// rustfmt-{}", config_name))),
                "config option file {} does not contain expected config name",
                path.display()
            );
        }
    }
}

#[test]
fn verify_config_test_names() {
    init_log();
    for path in &[
        Path::new("tests/source/configs"),
        Path::new("tests/target/configs"),
    ] {
        for entry in fs::read_dir(path).expect("couldn't read configs directory") {
            let entry = entry.expect("couldn't get directory entry");
            let path = entry.path();
            if path.is_dir() {
                let config_name = path.file_name().unwrap().to_str().unwrap();

                // Make sure that config name is used in the files in the directory.
                verify_config_used(&path, config_name);
            }
        }
    }
}

// This writes to the terminal using the same approach (via `term::stdout` or
// `println!`) that is used by `rustfmt::rustfmt_diff::print_diff`. Writing
// using only one or the other will cause the output order to differ when
// `print_diff` selects the approach not used.
fn write_message(msg: &str) {
    let mut writer = OutputWriter::new(Color::Auto);
    writer.writeln(msg, None);
}

// Integration tests. The files in `tests/source` are formatted and compared
// to their equivalent in `tests/target`. The target file and config can be
// overridden by annotations in the source file. The input and output must match
// exactly.
#[test]
fn system_tests() {
    init_log();
    run_test_with(&TestSetting::default(), || {
        // Get all files in the tests/source directory.
        let files = get_test_files(Path::new("tests/source"), true);
        let (_reports, count, fails) = check_files(files, &None);

        // Display results.
        println!("Ran {} system tests.", count);
        assert_eq!(fails, 0, "{} system tests failed", fails);
        assert!(
            count >= 300,
            "Expected a minimum of {} system tests to be executed",
            300
        )
    });
}

// Do the same for tests/coverage-source directory.
// The only difference is the coverage mode.
#[test]
fn coverage_tests() {
    init_log();
    let files = get_test_files(Path::new("tests/coverage/source"), true);
    let (_reports, count, fails) = check_files(files, &None);

    println!("Ran {} tests in coverage mode.", count);
    assert_eq!(fails, 0, "{} tests failed", fails);
}

#[test]
fn checkstyle_test() {
    init_log();
    let filename = "tests/writemode/source/fn-single-line.rs";
    let expected_filename = "tests/writemode/target/checkstyle.xml";
    assert_output(Path::new(filename), Path::new(expected_filename));
}

#[test]
fn json_test() {
    init_log();
    let filename = "tests/writemode/source/json.rs";
    let expected_filename = "tests/writemode/target/output.json";
    assert_output(Path::new(filename), Path::new(expected_filename));
}

#[test]
fn modified_test() {
    init_log();
    use std::io::BufRead;

    // Test "modified" output
    let filename = "tests/writemode/source/modified.rs";
    let mut data = Vec::new();
    let mut config = Config::default();
    config
        .set()
        .emit_mode(crate::config::EmitMode::ModifiedLines);

    {
        let mut session = Session::new(config, Some(&mut data));
        session.format(Input::File(filename.into())).unwrap();
    }

    let mut lines = data.lines();
    let mut chunks = Vec::new();
    while let Some(Ok(header)) = lines.next() {
        // Parse the header line
        let values: Vec<_> = header
            .split(' ')
            .map(|s| s.parse::<u32>().unwrap())
            .collect();
        assert_eq!(values.len(), 3);
        let line_number_orig = values[0];
        let lines_removed = values[1];
        let num_added = values[2];
        let mut added_lines = Vec::new();
        for _ in 0..num_added {
            added_lines.push(lines.next().unwrap().unwrap());
        }
        chunks.push(ModifiedChunk {
            line_number_orig,
            lines_removed,
            lines: added_lines,
        });
    }

    assert_eq!(
        chunks,
        vec![
            ModifiedChunk {
                line_number_orig: 4,
                lines_removed: 4,
                lines: vec!["fn blah() {}".into()],
            },
            ModifiedChunk {
                line_number_orig: 9,
                lines_removed: 6,
                lines: vec!["#[cfg(a, b)]".into(), "fn main() {}".into()],
            },
        ],
    );
}

// Helper function for comparing the results of rustfmt
// to a known output file generated by one of the write modes.
fn assert_output(source: &Path, expected_filename: &Path) {
    let config = read_config(source);
    let (_, source_file, _) = format_file(source, config.clone());

    // Populate output by writing to a vec.
    let mut out = vec![];
    let _ = source_file::write_all_files(&source_file, &mut out, &config);
    let output = String::from_utf8(out).unwrap();

    let mut expected_file = fs::File::open(&expected_filename).expect("couldn't open target");
    let mut expected_text = String::new();
    expected_file
        .read_to_string(&mut expected_text)
        .expect("Failed reading target");

    let compare = make_diff(&expected_text, &output, DIFF_CONTEXT_SIZE);
    if !compare.is_empty() {
        let mut failures = HashMap::new();
        failures.insert(source.to_owned(), compare);
        print_mismatches_default_message(failures);
        panic!("Text does not match expected output");
    }
}

// Idempotence tests. Files in tests/target are checked to be unaltered by
// rustfmt.
#[nightly_only_test]
#[test]
fn idempotence_tests() {
    init_log();
    run_test_with(&TestSetting::default(), || {
        // Get all files in the tests/target directory.
        let files = get_test_files(Path::new("tests/target"), true);
        let (_reports, count, fails) = check_files(files, &None);

        // Display results.
        println!("Ran {} idempotent tests.", count);
        assert_eq!(fails, 0, "{} idempotent tests failed", fails);
        assert!(
            count >= 400,
            "Expected a minimum of {} idempotent tests to be executed",
            400
        )
    });
}

// Run rustfmt on itself. This operation must be idempotent. We also check that
// no warnings are emitted.
// Issue-3443: these tests require nightly
#[nightly_only_test]
#[test]
fn self_tests() {
    init_log();
    let mut files = get_test_files(Path::new("tests"), false);
    let bin_directories = vec!["cargo-fmt", "git-rustfmt", "bin", "format-diff"];
    for dir in bin_directories {
        let mut path = PathBuf::from("src");
        path.push(dir);
        path.push("main.rs");
        files.push(path);
    }
    files.push(PathBuf::from("src/lib.rs"));

    let (reports, count, fails) = check_files(files, &Some(PathBuf::from("rustfmt.toml")));
    let mut warnings = 0;

    // Display results.
    println!("Ran {} self tests.", count);
    assert_eq!(fails, 0, "{} self tests failed", fails);

    for format_report in reports {
        println!(
            "{}",
            FormatReportFormatterBuilder::new(&format_report).build()
        );
        warnings += format_report.warning_count();
    }

    assert_eq!(
        warnings, 0,
        "Rustfmt's code generated {} warnings",
        warnings
    );
}

#[test]
fn format_files_find_new_files_via_cfg_if() {
    init_log();
    run_test_with(&TestSetting::default(), || {
        // To repro issue-4656, it is necessary that these files are parsed
        // as a part of the same session (hence this separate test runner).
        let files = vec![
            Path::new("tests/source/issue-4656/lib2.rs"),
            Path::new("tests/source/issue-4656/lib.rs"),
        ];

        let config = Config::default();
        let mut session = Session::<io::Stdout>::new(config, None);

        let mut write_result = HashMap::new();
        for file in files {
            assert!(file.exists());
            let result = session.format(Input::File(file.into())).unwrap();
            assert!(!session.has_formatting_errors());
            assert!(!result.has_warnings());
            let mut source_file = SourceFile::new();
            mem::swap(&mut session.source_file, &mut source_file);

            for (filename, text) in source_file {
                if let FileName::Real(ref filename) = filename {
                    write_result.insert(filename.to_owned(), text);
                }
            }
        }
        assert_eq!(
            3,
            write_result.len(),
            "Should have uncovered an extra file (format_me_please.rs) via lib.rs"
        );
        assert!(handle_result(write_result, None).is_ok());
    });
}

#[test]
fn stdin_formatting_smoke_test() {
    init_log();
    let input = Input::Text("fn main () {}".to_owned());
    let mut config = Config::default();
    config.set().emit_mode(EmitMode::Stdout);
    let mut buf: Vec<u8> = vec![];
    {
        let mut session = Session::new(config, Some(&mut buf));
        session.format(input).unwrap();
        assert!(session.has_no_errors());
    }

    #[cfg(not(windows))]
    assert_eq!(buf, "stdin:\n\nfn main() {}\n".as_bytes());
    #[cfg(windows)]
    assert_eq!(buf, "stdin:\n\nfn main() {}\r\n".as_bytes());
}

#[test]
fn stdin_parser_panic_caught() {
    init_log();
    // See issue #3239.
    for text in ["{", "}"].iter().cloned().map(String::from) {
        let mut buf = vec![];
        let mut session = Session::new(Default::default(), Some(&mut buf));
        let _ = session.format(Input::Text(text));

        assert!(session.has_parsing_errors());
    }
}

/// Ensures that `EmitMode::ModifiedLines` works with input from `stdin`. Useful
/// when embedding Rustfmt (e.g. inside RLS).
#[test]
fn stdin_works_with_modified_lines() {
    init_log();
    let input = "\nfn\n some( )\n{\n}\nfn main () {}\n";
    let output = "1 6 2\nfn some() {}\nfn main() {}\n";

    let input = Input::Text(input.to_owned());
    let mut config = Config::default();
    config.set().newline_style(NewlineStyle::Unix);
    config.set().emit_mode(EmitMode::ModifiedLines);
    let mut buf: Vec<u8> = vec![];
    {
        let mut session = Session::new(config, Some(&mut buf));
        session.format(input).unwrap();
        let errors = ReportedErrors {
            has_diff: true,
            ..Default::default()
        };
        assert_eq!(session.errors, errors);
    }
    assert_eq!(buf, output.as_bytes());
}

#[test]
fn stdin_disable_all_formatting_test() {
    init_log();
    let input = String::from("fn main() { println!(\"This should not be formatted.\"); }");
    let mut child = Command::new(rustfmt().to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .arg("--config-path=./tests/config/disable_all_formatting.toml")
        .spawn()
        .expect("failed to execute child");

    {
        let stdin = child.stdin.as_mut().expect("failed to get stdin");
        stdin
            .write_all(input.as_bytes())
            .expect("failed to write stdin");
    }

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(input, String::from_utf8(output.stdout).unwrap());
}

#[test]
fn stdin_generated_files_issue_5172() {
    init_log();
    let input = Input::Text("//@generated\nfn   main() {}".to_owned());
    let mut config = Config::default();
    config.set().emit_mode(EmitMode::Stdout);
    config.set().format_generated_files(false);
    config.set().newline_style(NewlineStyle::Unix);
    let mut buf: Vec<u8> = vec![];
    {
        let mut session = Session::new(config, Some(&mut buf));
        session.format(input).unwrap();
        assert!(session.has_no_errors());
    }
    // N.B. this should be changed once `format_generated_files` is supported with stdin
    assert_eq!(buf, "stdin:\n\n//@generated\nfn main() {}\n".as_bytes());
}

#[test]
fn format_lines_errors_are_reported() {
    init_log();
    let long_identifier = String::from_utf8(vec![b'a'; 239]).unwrap();
    let input = Input::Text(format!("fn {}() {{}}", long_identifier));
    let mut config = Config::default();
    config.set().error_on_line_overflow(true);
    let mut session = Session::<io::Stdout>::new(config, None);
    session.format(input).unwrap();
    assert!(session.has_formatting_errors());
}

#[test]
fn format_lines_errors_are_reported_with_tabs() {
    init_log();
    let long_identifier = String::from_utf8(vec![b'a'; 97]).unwrap();
    let input = Input::Text(format!("fn a() {{\n\t{}\n}}", long_identifier));
    let mut config = Config::default();
    config.set().error_on_line_overflow(true);
    config.set().hard_tabs(true);
    let mut session = Session::<io::Stdout>::new(config, None);
    session.format(input).unwrap();
    assert!(session.has_formatting_errors());
}

// For each file, run rustfmt and collect the output.
// Returns the number of files checked and the number of failures.
fn check_files(files: Vec<PathBuf>, opt_config: &Option<PathBuf>) -> (Vec<FormatReport>, u32, u32) {
    let mut count = 0;
    let mut fails = 0;
    let mut reports = vec![];

    for file_name in files {
        let sig_comments = read_significant_comments(&file_name);
        if sig_comments.contains_key("unstable") && !is_nightly_channel!() {
            debug!(
                "Skipping '{}' because it requires unstable \
                 features which are only available on nightly...",
                file_name.display()
            );
            continue;
        }

        debug!("Testing '{}'...", file_name.display());

        match idempotent_check(&file_name, opt_config) {
            Ok(ref report) if report.has_warnings() => {
                print!("{}", FormatReportFormatterBuilder::new(report).build());
                fails += 1;
            }
            Ok(report) => reports.push(report),
            Err(err) => {
                if let IdempotentCheckError::Mismatch(msg) = err {
                    print_mismatches_default_message(msg);
                }
                fails += 1;
            }
        }

        count += 1;
    }

    (reports, count, fails)
}

fn print_mismatches_default_message(result: HashMap<PathBuf, Vec<Mismatch>>) {
    for (file_name, diff) in result {
        let mismatch_msg_formatter =
            |line_num| format!("\nMismatch at {}:{}:", file_name.display(), line_num);
        print_diff(diff, &mismatch_msg_formatter, &Default::default());
    }

    if let Some(mut t) = term::stdout() {
        t.reset().unwrap_or(());
    }
}

fn print_mismatches<T: Fn(u32) -> String>(
    result: HashMap<PathBuf, Vec<Mismatch>>,
    mismatch_msg_formatter: T,
) {
    for (_file_name, diff) in result {
        print_diff(diff, &mismatch_msg_formatter, &Default::default());
    }

    if let Some(mut t) = term::stdout() {
        t.reset().unwrap_or(());
    }
}

fn read_config(filename: &Path) -> Config {
    let sig_comments = read_significant_comments(filename);
    // Look for a config file. If there is a 'config' property in the significant comments, use
    // that. Otherwise, if there are no significant comments at all, look for a config file with
    // the same name as the test file.
    let mut config = if !sig_comments.is_empty() {
        get_config(sig_comments.get("config").map(Path::new))
    } else {
        get_config(filename.with_extension("toml").file_name().map(Path::new))
    };

    for (key, val) in &sig_comments {
        if key != "target" && key != "config" && key != "unstable" {
            config.override_value(key, val);
            if config.is_default(key) {
                warn!("Default value {} used explicitly for {}", val, key);
            }
        }
    }

    // Don't generate warnings for to-do items.
    config.set().report_todo(ReportTactic::Never);

    config
}

fn format_file<P: Into<PathBuf>>(filepath: P, config: Config) -> (bool, SourceFile, FormatReport) {
    let filepath = filepath.into();
    let input = Input::File(filepath);
    let mut session = Session::<io::Stdout>::new(config, None);
    let result = session.format(input).unwrap();
    let parsing_errors = session.has_parsing_errors();
    let mut source_file = SourceFile::new();
    mem::swap(&mut session.source_file, &mut source_file);
    (parsing_errors, source_file, result)
}

enum IdempotentCheckError {
    Mismatch(HashMap<PathBuf, Vec<Mismatch>>),
    Parse,
}

fn idempotent_check(
    filename: &PathBuf,
    opt_config: &Option<PathBuf>,
) -> Result<FormatReport, IdempotentCheckError> {
    let sig_comments = read_significant_comments(filename);
    let config = if let Some(ref config_file_path) = opt_config {
        Config::from_toml_path(config_file_path).expect("`rustfmt.toml` not found")
    } else {
        read_config(filename)
    };
    let (parsing_errors, source_file, format_report) = format_file(filename, config);
    if parsing_errors {
        return Err(IdempotentCheckError::Parse);
    }

    let mut write_result = HashMap::new();
    for (filename, text) in source_file {
        if let FileName::Real(ref filename) = filename {
            write_result.insert(filename.to_owned(), text);
        }
    }

    let target = sig_comments.get("target").map(|x| &(*x)[..]);

    handle_result(write_result, target).map(|_| format_report)
}

// Reads test config file using the supplied (optional) file name. If there's no file name or the
// file doesn't exist, just return the default config. Otherwise, the file must be read
// successfully.
fn get_config(config_file: Option<&Path>) -> Config {
    let config_file_name = match config_file {
        None => return Default::default(),
        Some(file_name) => {
            let mut full_path = PathBuf::from("tests/config/");
            full_path.push(file_name);
            if !full_path.exists() {
                return Default::default();
            };
            full_path
        }
    };

    let mut def_config_file = fs::File::open(config_file_name).expect("couldn't open config");
    let mut def_config = String::new();
    def_config_file
        .read_to_string(&mut def_config)
        .expect("Couldn't read config");

    Config::from_toml(&def_config, Path::new("tests/config/")).expect("invalid TOML")
}

// Reads significant comments of the form: `// rustfmt-key: value` into a hash map.
fn read_significant_comments(file_name: &Path) -> HashMap<String, String> {
    let file = fs::File::open(file_name)
        .unwrap_or_else(|_| panic!("couldn't read file {}", file_name.display()));
    let reader = BufReader::new(file);
    let pattern = r"^\s*//\s*rustfmt-([^:]+):\s*(\S+)";
    let regex = regex::Regex::new(pattern).expect("failed creating pattern 1");

    // Matches lines containing significant comments or whitespace.
    let line_regex = regex::Regex::new(r"(^\s*$)|(^\s*//\s*rustfmt-[^:]+:\s*\S+)")
        .expect("failed creating pattern 2");

    reader
        .lines()
        .map(|line| line.expect("failed getting line"))
        .filter(|line| line_regex.is_match(line))
        .filter_map(|line| {
            regex.captures_iter(&line).next().map(|capture| {
                (
                    capture
                        .get(1)
                        .expect("couldn't unwrap capture")
                        .as_str()
                        .to_owned(),
                    capture
                        .get(2)
                        .expect("couldn't unwrap capture")
                        .as_str()
                        .to_owned(),
                )
            })
        })
        .collect()
}

// Compares output to input.
// TODO: needs a better name, more explanation.
fn handle_result(
    result: HashMap<PathBuf, String>,
    target: Option<&str>,
) -> Result<(), IdempotentCheckError> {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        // If file is in tests/source, compare to file with same name in tests/target.
        let target = get_target(&file_name, target);
        let open_error = format!("couldn't open target {:?}", target);
        let mut f = fs::File::open(&target).expect(&open_error);

        let mut text = String::new();
        let read_error = format!("failed reading target {:?}", target);
        f.read_to_string(&mut text).expect(&read_error);

        // Ignore LF and CRLF difference for Windows.
        if !string_eq_ignore_newline_repr(&fmt_text, &text) {
            let diff = make_diff(&text, &fmt_text, DIFF_CONTEXT_SIZE);
            assert!(
                !diff.is_empty(),
                "Empty diff? Maybe due to a missing a newline at the end of a file?"
            );
            failures.insert(file_name, diff);
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(IdempotentCheckError::Mismatch(failures))
    }
}

// Maps source file paths to their target paths.
fn get_target(file_name: &Path, target: Option<&str>) -> PathBuf {
    if let Some(n) = file_name
        .components()
        .position(|c| c.as_os_str() == "source")
    {
        let mut target_file_name = PathBuf::new();
        for (i, c) in file_name.components().enumerate() {
            if i == n {
                target_file_name.push("target");
            } else {
                target_file_name.push(c.as_os_str());
            }
        }
        if let Some(replace_name) = target {
            target_file_name.with_file_name(replace_name)
        } else {
            target_file_name
        }
    } else {
        // This is either and idempotence check or a self check.
        file_name.to_owned()
    }
}

#[test]
fn rustfmt_diff_make_diff_tests() {
    init_log();
    let diff = make_diff("a\nb\nc\nd", "a\ne\nc\nd", 3);
    assert_eq!(
        diff,
        vec![Mismatch {
            line_number: 1,
            line_number_orig: 1,
            lines: vec![
                DiffLine::Context("a".into()),
                DiffLine::Resulting("b".into()),
                DiffLine::Expected("e".into()),
                DiffLine::Context("c".into()),
                DiffLine::Context("d".into()),
            ],
        }]
    );
}

#[test]
fn rustfmt_diff_no_diff_test() {
    init_log();
    let diff = make_diff("a\nb\nc\nd", "a\nb\nc\nd", 3);
    assert_eq!(diff, vec![]);
}

// Compare strings without distinguishing between CRLF and LF
fn string_eq_ignore_newline_repr(left: &str, right: &str) -> bool {
    let left = CharsIgnoreNewlineRepr(left.chars().peekable());
    let right = CharsIgnoreNewlineRepr(right.chars().peekable());
    left.eq(right)
}

struct CharsIgnoreNewlineRepr<'a>(Peekable<Chars<'a>>);

impl<'a> Iterator for CharsIgnoreNewlineRepr<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        self.0.next().map(|c| {
            if c == '\r' {
                if *self.0.peek().unwrap_or(&'\0') == '\n' {
                    self.0.next();
                    '\n'
                } else {
                    '\r'
                }
            } else {
                c
            }
        })
    }
}

#[test]
fn string_eq_ignore_newline_repr_test() {
    init_log();
    assert!(string_eq_ignore_newline_repr("", ""));
    assert!(!string_eq_ignore_newline_repr("", "abc"));
    assert!(!string_eq_ignore_newline_repr("abc", ""));
    assert!(string_eq_ignore_newline_repr("a\nb\nc\rd", "a\nb\r\nc\rd"));
    assert!(string_eq_ignore_newline_repr("a\r\n\r\n\r\nb", "a\n\n\nb"));
    assert!(!string_eq_ignore_newline_repr("a\r\nbcd", "a\nbcdefghijk"));
}

struct TempFile {
    path: PathBuf,
}

fn make_temp_file(file_name: &'static str) -> TempFile {
    use std::env::var;
    use std::fs::File;

    // Used in the Rust build system.
    let target_dir = var("RUSTFMT_TEST_DIR").unwrap_or_else(|_| ".".to_owned());
    let path = Path::new(&target_dir).join(file_name);

    let mut file = File::create(&path).expect("couldn't create temp file");
    let content = "fn main() {}\n";
    file.write_all(content.as_bytes())
        .expect("couldn't write temp file");
    TempFile { path }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        use std::fs::remove_file;
        remove_file(&self.path).expect("couldn't delete temp file");
    }
}

fn rustfmt() -> PathBuf {
    let mut me = env::current_exe().expect("failed to get current executable");
    // Chop of the test name.
    me.pop();
    // Chop off `deps`.
    me.pop();

    // If we run `cargo test --release`, we might only have a release build.
    if cfg!(release) {
        // `../release/`
        me.pop();
        me.push("release");
    }
    me.push("rustfmt");
    assert!(
        me.is_file() || me.with_extension("exe").is_file(),
        "{}",
        if cfg!(release) {
            "no rustfmt bin, try running `cargo build --release` before testing"
        } else {
            "no rustfmt bin, try running `cargo build` before testing"
        }
    );
    me
}

#[test]
fn verify_check_works() {
    init_log();
    let temp_file = make_temp_file("temp_check.rs");

    Command::new(rustfmt().to_str().unwrap())
        .arg("--check")
        .arg(temp_file.path.to_str().unwrap())
        .status()
        .expect("run with check option failed");
}
