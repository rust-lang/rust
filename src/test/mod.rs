// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate assert_cli;

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::iter::{Enumerate, Peekable};
use std::mem;
use std::path::{Path, PathBuf};
use std::str::Chars;

use config::{Color, Config, EmitMode, FileName, ReportTactic};
use formatting::{ModifiedChunk, SourceFile};
use rustfmt_diff::{make_diff, print_diff, DiffLine, Mismatch, OutputWriter};
use source_file;
use {FormatReport, Input, Session};

const DIFF_CONTEXT_SIZE: usize = 3;
const CONFIGURATIONS_FILE_NAME: &str = "Configurations.md";

// Returns a `Vec` containing `PathBuf`s of files with a rs extension in the
// given path. The `recursive` argument controls if files from subdirectories
// are also returned.
fn get_test_files(path: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut files = vec![];
    if path.is_dir() {
        for entry in fs::read_dir(path).expect(&format!(
            "Couldn't read directory {}",
            path.to_str().unwrap()
        )) {
            let entry = entry.expect("Couldn't get DirEntry");
            let path = entry.path();
            if path.is_dir() && recursive {
                files.append(&mut get_test_files(&path, recursive));
            } else if path.extension().map_or(false, |f| f == "rs") {
                files.push(path);
            }
        }
    }
    files
}

fn verify_config_used(path: &Path, config_name: &str) {
    for entry in fs::read_dir(path).expect(&format!(
        "Couldn't read {} directory",
        path.to_str().unwrap()
    )) {
        let entry = entry.expect("Couldn't get directory entry");
        let path = entry.path();
        if path.extension().map_or(false, |f| f == "rs") {
            // check if "// rustfmt-<config_name>:" appears in the file.
            let filebuf = BufReader::new(
                fs::File::open(&path).expect(&format!("Couldn't read file {}", path.display())),
            );
            assert!(
                filebuf
                    .lines()
                    .map(|l| l.unwrap())
                    .take_while(|l| l.starts_with("//"))
                    .any(|l| l.starts_with(&format!("// rustfmt-{}", config_name))),
                format!(
                    "config option file {} does not contain expected config name",
                    path.display()
                )
            );
        }
    }
}

#[test]
fn verify_config_test_names() {
    for path in &[
        Path::new("tests/source/configs"),
        Path::new("tests/target/configs"),
    ] {
        for entry in fs::read_dir(path).expect("Couldn't read configs directory") {
            let entry = entry.expect("Couldn't get directory entry");
            let path = entry.path();
            if path.is_dir() {
                let config_name = path.file_name().unwrap().to_str().unwrap();

                // Make sure that config name is used in the files in the directory.
                verify_config_used(&path, config_name);
            }
        }
    }
}

// This writes to the terminal using the same approach (via term::stdout or
// println!) that is used by `rustfmt::rustfmt_diff::print_diff`. Writing
// using only one or the other will cause the output order to differ when
// `print_diff` selects the approach not used.
fn write_message(msg: &str) {
    let mut writer = OutputWriter::new(Color::Auto);
    writer.writeln(&format!("{}", msg), None);
}

// Integration tests. The files in the tests/source are formatted and compared
// to their equivalent in tests/target. The target file and config can be
// overridden by annotations in the source file. The input and output must match
// exactly.
#[test]
fn system_tests() {
    // Get all files in the tests/source directory.
    let files = get_test_files(Path::new("tests/source"), true);
    let (_reports, count, fails) = check_files(files, None);

    // Display results.
    println!("Ran {} system tests.", count);
    assert_eq!(fails, 0, "{} system tests failed", fails);
}

// Do the same for tests/coverage-source directory
// the only difference is the coverage mode
#[test]
fn coverage_tests() {
    let files = get_test_files(Path::new("tests/coverage/source"), true);
    let (_reports, count, fails) = check_files(files, None);

    println!("Ran {} tests in coverage mode.", count);
    assert_eq!(fails, 0, "{} tests failed", fails);
}

#[test]
fn checkstyle_test() {
    let filename = "tests/writemode/source/fn-single-line.rs";
    let expected_filename = "tests/writemode/target/checkstyle.xml";
    assert_output(Path::new(filename), Path::new(expected_filename));
}

#[test]
fn modified_test() {
    use std::io::BufRead;

    // Test "modified" output
    let filename = "tests/writemode/source/modified.rs";
    let mut data = Vec::new();
    let mut config = Config::default();
    config.set().emit_mode(::config::EmitMode::ModifiedLines);

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

    let mut expected_file = fs::File::open(&expected_filename).expect("Couldn't open target");
    let mut expected_text = String::new();
    expected_file
        .read_to_string(&mut expected_text)
        .expect("Failed reading target");

    let compare = make_diff(&expected_text, &output, DIFF_CONTEXT_SIZE);
    if !compare.is_empty() {
        let mut failures = HashMap::new();
        failures.insert(source.to_owned(), compare);
        print_mismatches_default_message(failures);
        assert!(false, "Text does not match expected output");
    }
}

// Idempotence tests. Files in tests/target are checked to be unaltered by
// rustfmt.
#[test]
fn idempotence_tests() {
    match option_env!("CFG_RELEASE_CHANNEL") {
        None | Some("nightly") => {}
        _ => return, // these tests require nightly
    }
    // Get all files in the tests/target directory.
    let files = get_test_files(Path::new("tests/target"), true);
    let (_reports, count, fails) = check_files(files, None);

    // Display results.
    println!("Ran {} idempotent tests.", count);
    assert_eq!(fails, 0, "{} idempotent tests failed", fails);
}

// Run rustfmt on itself. This operation must be idempotent. We also check that
// no warnings are emitted.
#[test]
fn self_tests() {
    let mut files = get_test_files(Path::new("tests"), false);
    let bin_directories = vec!["cargo-fmt", "git-rustfmt", "bin", "format-diff"];
    for dir in bin_directories {
        let mut path = PathBuf::from("src");
        path.push(dir);
        path.push("main.rs");
        files.push(path);
    }
    files.push(PathBuf::from("src/lib.rs"));

    let (reports, count, fails) = check_files(files, Some(PathBuf::from("rustfmt.toml")));
    let mut warnings = 0;

    // Display results.
    println!("Ran {} self tests.", count);
    assert_eq!(fails, 0, "{} self tests failed", fails);

    for format_report in reports {
        println!("{}", format_report);
        warnings += format_report.warning_count();
    }

    assert_eq!(
        warnings, 0,
        "Rustfmt's code generated {} warnings",
        warnings
    );
}

#[test]
fn stdin_formatting_smoke_test() {
    let input = Input::Text("fn main () {}".to_owned());
    let mut config = Config::default();
    config.set().emit_mode(EmitMode::Stdout);
    let mut buf: Vec<u8> = vec![];
    {
        let mut session = Session::new(config, Some(&mut buf));
        session.format(input).unwrap();
        assert!(session.has_no_errors());
    }
    //eprintln!("{:?}", );
    #[cfg(not(windows))]
    assert_eq!(buf, "fn main() {}\n".as_bytes());
    #[cfg(windows)]
    assert_eq!(buf, "fn main() {}\r\n".as_bytes());
}

// FIXME(#1990) restore this test
// #[test]
// fn stdin_disable_all_formatting_test() {
//     let input = String::from("fn main() { println!(\"This should not be formatted.\"); }");
//     let mut child = Command::new("./target/debug/rustfmt")
//         .stdin(Stdio::piped())
//         .stdout(Stdio::piped())
//         .arg("--config-path=./tests/config/disable_all_formatting.toml")
//         .spawn()
//         .expect("failed to execute child");

//     {
//         let stdin = child.stdin.as_mut().expect("failed to get stdin");
//         stdin
//             .write_all(input.as_bytes())
//             .expect("failed to write stdin");
//     }
//     let output = child.wait_with_output().expect("failed to wait on child");
//     assert!(output.status.success());
//     assert!(output.stderr.is_empty());
//     assert_eq!(input, String::from_utf8(output.stdout).unwrap());
// }

#[test]
fn format_lines_errors_are_reported() {
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
fn check_files(files: Vec<PathBuf>, opt_config: Option<PathBuf>) -> (Vec<FormatReport>, u32, u32) {
    let mut count = 0;
    let mut fails = 0;
    let mut reports = vec![];

    for file_name in files {
        debug!("Testing '{}'...", file_name.display());

        match idempotent_check(&file_name, &opt_config) {
            Ok(ref report) if report.has_warnings() => {
                print!("{}", report);
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
    // Look for a config file... If there is a 'config' property in the significant comments, use
    // that. Otherwise, if there are no significant comments at all, look for a config file with
    // the same name as the test file.
    let mut config = if !sig_comments.is_empty() {
        get_config(sig_comments.get("config").map(Path::new))
    } else {
        get_config(filename.with_extension("toml").file_name().map(Path::new))
    };

    for (key, val) in &sig_comments {
        if key != "target" && key != "config" {
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
        Config::from_toml_path(config_file_path).expect("rustfmt.toml not found")
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

    let mut def_config_file = fs::File::open(config_file_name).expect("Couldn't open config");
    let mut def_config = String::new();
    def_config_file
        .read_to_string(&mut def_config)
        .expect("Couldn't read config");

    Config::from_toml(&def_config, Path::new("tests/config/")).expect("Invalid toml")
}

// Reads significant comments of the form: // rustfmt-key: value
// into a hash map.
fn read_significant_comments(file_name: &Path) -> HashMap<String, String> {
    let file =
        fs::File::open(file_name).expect(&format!("Couldn't read file {}", file_name.display()));
    let reader = BufReader::new(file);
    let pattern = r"^\s*//\s*rustfmt-([^:]+):\s*(\S+)";
    let regex = regex::Regex::new(pattern).expect("Failed creating pattern 1");

    // Matches lines containing significant comments or whitespace.
    let line_regex = regex::Regex::new(r"(^\s*$)|(^\s*//\s*rustfmt-[^:]+:\s*\S+)")
        .expect("Failed creating pattern 2");

    reader
        .lines()
        .map(|line| line.expect("Failed getting line"))
        .take_while(|line| line_regex.is_match(line))
        .filter_map(|line| {
            regex.captures_iter(&line).next().map(|capture| {
                (
                    capture
                        .get(1)
                        .expect("Couldn't unwrap capture")
                        .as_str()
                        .to_owned(),
                    capture
                        .get(2)
                        .expect("Couldn't unwrap capture")
                        .as_str()
                        .to_owned(),
                )
            })
        }).collect()
}

// Compare output to input.
// TODO: needs a better name, more explanation.
fn handle_result(
    result: HashMap<PathBuf, String>,
    target: Option<&str>,
) -> Result<(), IdempotentCheckError> {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        // If file is in tests/source, compare to file with same name in tests/target.
        let target = get_target(&file_name, target);
        let open_error = format!("Couldn't open target {:?}", &target);
        let mut f = fs::File::open(&target).expect(&open_error);

        let mut text = String::new();
        let read_error = format!("Failed reading target {:?}", &target);
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

// Map source file paths to their target paths.
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
        // This is either and idempotence check or a self check
        file_name.to_owned()
    }
}

#[test]
fn rustfmt_diff_make_diff_tests() {
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
    assert!(string_eq_ignore_newline_repr("", ""));
    assert!(!string_eq_ignore_newline_repr("", "abc"));
    assert!(!string_eq_ignore_newline_repr("abc", ""));
    assert!(string_eq_ignore_newline_repr("a\nb\nc\rd", "a\nb\r\nc\rd"));
    assert!(string_eq_ignore_newline_repr("a\r\n\r\n\r\nb", "a\n\n\nb"));
    assert!(!string_eq_ignore_newline_repr("a\r\nbcd", "a\nbcdefghijk"));
}

// This enum is used to represent one of three text features in Configurations.md: a block of code
// with its starting line number, the name of a rustfmt configuration option, or the value of a
// rustfmt configuration option.
enum ConfigurationSection {
    CodeBlock((String, u32)), // (String: block of code, u32: line number of code block start)
    ConfigName(String),
    ConfigValue(String),
}

impl ConfigurationSection {
    fn get_section<I: Iterator<Item = String>>(
        file: &mut Enumerate<I>,
    ) -> Option<ConfigurationSection> {
        lazy_static! {
            static ref CONFIG_NAME_REGEX: regex::Regex =
                regex::Regex::new(r"^## `([^`]+)`").expect("Failed creating configuration pattern");
            static ref CONFIG_VALUE_REGEX: regex::Regex =
                regex::Regex::new(r#"^#### `"?([^`"]+)"?`"#)
                    .expect("Failed creating configuration value pattern");
        }

        loop {
            match file.next() {
                Some((i, line)) => {
                    if line.starts_with("```rust") {
                        // Get the lines of the code block.
                        let lines: Vec<String> = file
                            .map(|(_i, l)| l)
                            .take_while(|l| !l.starts_with("```"))
                            .collect();
                        let block = format!("{}\n", lines.join("\n"));

                        // +1 to translate to one-based indexing
                        // +1 to get to first line of code (line after "```")
                        let start_line = (i + 2) as u32;

                        return Some(ConfigurationSection::CodeBlock((block, start_line)));
                    } else if let Some(c) = CONFIG_NAME_REGEX.captures(&line) {
                        return Some(ConfigurationSection::ConfigName(String::from(&c[1])));
                    } else if let Some(c) = CONFIG_VALUE_REGEX.captures(&line) {
                        return Some(ConfigurationSection::ConfigValue(String::from(&c[1])));
                    }
                }
                None => return None, // reached the end of the file
            }
        }
    }
}

// This struct stores the information about code blocks in the configurations
// file, formats the code blocks, and prints formatting errors.
struct ConfigCodeBlock {
    config_name: Option<String>,
    config_value: Option<String>,
    code_block: Option<String>,
    code_block_start: Option<u32>,
}

impl ConfigCodeBlock {
    fn new() -> ConfigCodeBlock {
        ConfigCodeBlock {
            config_name: None,
            config_value: None,
            code_block: None,
            code_block_start: None,
        }
    }

    fn set_config_name(&mut self, name: Option<String>) {
        self.config_name = name;
        self.config_value = None;
    }

    fn set_config_value(&mut self, value: Option<String>) {
        self.config_value = value;
    }

    fn set_code_block(&mut self, code_block: String, code_block_start: u32) {
        self.code_block = Some(code_block);
        self.code_block_start = Some(code_block_start);
    }

    fn get_block_config(&self) -> Config {
        let mut config = Config::default();
        if self.config_value.is_some() && self.config_value.is_some() {
            config.override_value(
                self.config_name.as_ref().unwrap(),
                self.config_value.as_ref().unwrap(),
            );
        }
        config
    }

    fn code_block_valid(&self) -> bool {
        // We never expect to not have a code block.
        assert!(self.code_block.is_some() && self.code_block_start.is_some());

        // See if code block begins with #![rustfmt::skip].
        let fmt_skip = self
            .code_block
            .as_ref()
            .unwrap()
            .split('\n')
            .nth(0)
            .unwrap_or("")
            == "#![rustfmt::skip]";

        if self.config_name.is_none() && !fmt_skip {
            write_message(&format!(
                "No configuration name for {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return false;
        }
        if self.config_value.is_none() && !fmt_skip {
            write_message(&format!(
                "No configuration value for {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return false;
        }
        true
    }

    fn has_parsing_errors<T: Write>(&self, session: &Session<T>) -> bool {
        if session.has_parsing_errors() {
            write_message(&format!(
                "\u{261d}\u{1f3fd} Cannot format {}:{}",
                CONFIGURATIONS_FILE_NAME,
                self.code_block_start.unwrap()
            ));
            return true;
        }

        false
    }

    fn print_diff(&self, compare: Vec<Mismatch>) {
        let mut mismatches = HashMap::new();
        mismatches.insert(PathBuf::from(CONFIGURATIONS_FILE_NAME), compare);
        print_mismatches(mismatches, |line_num| {
            format!(
                "\nMismatch at {}:{}:",
                CONFIGURATIONS_FILE_NAME,
                line_num + self.code_block_start.unwrap() - 1
            )
        });
    }

    fn formatted_has_diff(&self, text: &str) -> bool {
        let compare = make_diff(self.code_block.as_ref().unwrap(), text, DIFF_CONTEXT_SIZE);
        if !compare.is_empty() {
            self.print_diff(compare);
            return true;
        }

        false
    }

    // Return a bool indicating if formatting this code block is an idempotent
    // operation. This function also triggers printing any formatting failure
    // messages.
    fn formatted_is_idempotent(&self) -> bool {
        // Verify that we have all of the expected information.
        if !self.code_block_valid() {
            return false;
        }

        let input = Input::Text(self.code_block.as_ref().unwrap().to_owned());
        let mut config = self.get_block_config();
        config.set().emit_mode(EmitMode::Stdout);
        let mut buf: Vec<u8> = vec![];

        {
            let mut session = Session::new(config, Some(&mut buf));
            session.format(input).unwrap();
            if self.has_parsing_errors(&session) {
                return false;
            }
        }

        !self.formatted_has_diff(&String::from_utf8(buf).unwrap())
    }

    // Extract a code block from the iterator. Behavior:
    // - Rust code blocks are identifed by lines beginning with "```rust".
    // - One explicit configuration setting is supported per code block.
    // - Rust code blocks with no configuration setting are illegal and cause an
    //   assertion failure, unless the snippet begins with #![rustfmt::skip].
    // - Configuration names in Configurations.md must be in the form of
    //   "## `NAME`".
    // - Configuration values in Configurations.md must be in the form of
    //   "#### `VALUE`".
    fn extract<I: Iterator<Item = String>>(
        file: &mut Enumerate<I>,
        prev: Option<&ConfigCodeBlock>,
        hash_set: &mut HashSet<String>,
    ) -> Option<ConfigCodeBlock> {
        let mut code_block = ConfigCodeBlock::new();
        code_block.config_name = prev.and_then(|cb| cb.config_name.clone());

        loop {
            match ConfigurationSection::get_section(file) {
                Some(ConfigurationSection::CodeBlock((block, start_line))) => {
                    code_block.set_code_block(block, start_line);
                    break;
                }
                Some(ConfigurationSection::ConfigName(name)) => {
                    assert!(
                        Config::is_valid_name(&name),
                        "an unknown configuration option was found: {}",
                        name
                    );
                    assert!(
                        hash_set.remove(&name),
                        "multiple configuration guides found for option {}",
                        name
                    );
                    code_block.set_config_name(Some(name));
                }
                Some(ConfigurationSection::ConfigValue(value)) => {
                    code_block.set_config_value(Some(value));
                }
                None => return None, // end of file was reached
            }
        }

        Some(code_block)
    }
}

#[test]
fn configuration_snippet_tests() {
    // Read Configurations.md and build a `Vec` of `ConfigCodeBlock` structs with one
    // entry for each Rust code block found.
    fn get_code_blocks() -> Vec<ConfigCodeBlock> {
        let mut file_iter = BufReader::new(
            fs::File::open(Path::new(CONFIGURATIONS_FILE_NAME))
                .expect(&format!("Couldn't read file {}", CONFIGURATIONS_FILE_NAME)),
        ).lines()
        .map(|l| l.unwrap())
        .enumerate();
        let mut code_blocks: Vec<ConfigCodeBlock> = Vec::new();
        let mut hash_set = Config::hash_set();

        while let Some(cb) =
            ConfigCodeBlock::extract(&mut file_iter, code_blocks.last(), &mut hash_set)
        {
            code_blocks.push(cb);
        }

        for name in hash_set {
            if !Config::is_hidden_option(&name) {
                panic!("{} does not have a configuration guide", name);
            }
        }

        code_blocks
    }

    let blocks = get_code_blocks();
    let failures = blocks
        .iter()
        .map(|b| b.formatted_is_idempotent())
        .fold(0, |acc, r| acc + (!r as u32));

    // Display results.
    println!("Ran {} configurations tests.", blocks.len());
    assert_eq!(failures, 0, "{} configurations tests failed", failures);
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

    let mut file = File::create(&path).expect("Couldn't create temp file");
    let content = "fn main() {}\n";
    file.write_all(content.as_bytes())
        .expect("Couldn't write temp file");
    TempFile { path }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        use std::fs::remove_file;
        remove_file(&self.path).expect("Couldn't delete temp file");
    }
}

fn rustfmt() -> PathBuf {
    let mut me = env::current_exe().expect("failed to get current executable");
    me.pop(); // chop of the test name
    me.pop(); // chop off `deps`
    me.push("rustfmt");
    assert!(
        me.is_file() || me.with_extension("exe").is_file(),
        "no rustfmt bin, try running `cargo build` before testing"
    );
    return me;
}

#[test]
fn verify_check_works() {
    let temp_file = make_temp_file("temp_check.rs");
    assert_cli::Assert::command(&[
        rustfmt().to_str().unwrap(),
        "--check",
        temp_file.path.to_str().unwrap(),
    ]).succeeds()
    .unwrap();
}
