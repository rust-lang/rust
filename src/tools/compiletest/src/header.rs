// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

use common::Config;
use common;
use util;

use extract_gdb_version;

/// Properties which must be known very early, before actually running
/// the test.
pub struct EarlyProps {
    pub ignore: bool,
    pub should_fail: bool,
}

impl EarlyProps {
    pub fn from_file(config: &Config, testfile: &Path) -> Self {
        let mut props = EarlyProps {
            ignore: false,
            should_fail: false,
        };

        iter_header(testfile,
                    None,
                    &mut |ln| {
            props.ignore =
                props.ignore || parse_name_directive(ln, "ignore-test") ||
                parse_name_directive(ln, &ignore_target(config)) ||
                parse_name_directive(ln, &ignore_architecture(config)) ||
                parse_name_directive(ln, &ignore_stage(config)) ||
                parse_name_directive(ln, &ignore_env(config)) ||
                (config.mode == common::Pretty && parse_name_directive(ln, "ignore-pretty")) ||
                (config.target != config.host &&
                 parse_name_directive(ln, "ignore-cross-compile")) ||
                ignore_gdb(config, ln) ||
                ignore_lldb(config, ln) ||
                ignore_llvm(config, ln);

            props.should_fail = props.should_fail || parse_name_directive(ln, "should-fail");
        });

        return props;

        fn ignore_target(config: &Config) -> String {
            format!("ignore-{}", util::get_os(&config.target))
        }
        fn ignore_architecture(config: &Config) -> String {
            format!("ignore-{}", util::get_arch(&config.target))
        }
        fn ignore_stage(config: &Config) -> String {
            format!("ignore-{}", config.stage_id.split('-').next().unwrap())
        }
        fn ignore_env(config: &Config) -> String {
            format!("ignore-{}",
                    util::get_env(&config.target).unwrap_or("<unknown>"))
        }
        fn ignore_gdb(config: &Config, line: &str) -> bool {
            if config.mode != common::DebugInfoGdb {
                return false;
            }

            if !line.contains("ignore-gdb-version") &&
               parse_name_directive(line, "ignore-gdb") {
                return true;
            }

            if let Some(actual_version) = config.gdb_version {
                if line.contains("min-gdb-version") {
                    let (start_ver, end_ver) = extract_gdb_version_range(line);

                    if start_ver != end_ver {
                        panic!("Expected single GDB version")
                    }
                    // Ignore if actual version is smaller the minimum required
                    // version
                    actual_version < start_ver
                } else if line.contains("ignore-gdb-version") {
                    let (min_version, max_version) = extract_gdb_version_range(line);

                    if max_version < min_version {
                        panic!("Malformed GDB version range: max < min")
                    }

                    actual_version >= min_version && actual_version <= max_version
                } else {
                    false
                }
            } else {
                false
            }
        }

        // Takes a directive of the form "ignore-gdb-version <version1> [- <version2>]",
        // returns the numeric representation of <version1> and <version2> as
        // tuple: (<version1> as u32, <version2> as u32)
        // If the <version2> part is omitted, the second component of the tuple
        // is the same as <version1>.
        fn extract_gdb_version_range(line: &str) -> (u32, u32) {
            const ERROR_MESSAGE: &'static str = "Malformed GDB version directive";

            let range_components = line.split(' ')
                                       .flat_map(|word| word.split('-'))
                                       .filter(|word| word.len() > 0)
                                       .skip_while(|word| extract_gdb_version(word).is_none())
                                       .collect::<Vec<&str>>();

            match range_components.len() {
                1 => {
                    let v = extract_gdb_version(range_components[0]).unwrap();
                    (v, v)
                }
                2 => {
                    let v_min = extract_gdb_version(range_components[0]).unwrap();
                    let v_max = extract_gdb_version(range_components[1]).expect(ERROR_MESSAGE);
                    (v_min, v_max)
                }
                _ => panic!(ERROR_MESSAGE),
            }
        }

        fn ignore_lldb(config: &Config, line: &str) -> bool {
            if config.mode != common::DebugInfoLldb {
                return false;
            }

            if parse_name_directive(line, "ignore-lldb") {
                return true;
            }

            if let Some(ref actual_version) = config.lldb_version {
                if line.contains("min-lldb-version") {
                    let min_version = line.trim()
                        .split(' ')
                        .last()
                        .expect("Malformed lldb version directive");
                    // Ignore if actual version is smaller the minimum required
                    // version
                    lldb_version_to_int(actual_version) < lldb_version_to_int(min_version)
                } else {
                    false
                }
            } else {
                false
            }
        }

        fn ignore_llvm(config: &Config, line: &str) -> bool {
            if let Some(ref actual_version) = config.llvm_version {
                if line.contains("min-llvm-version") {
                    let min_version = line.trim()
                        .split(' ')
                        .last()
                        .expect("Malformed llvm version directive");
                    // Ignore if actual version is smaller the minimum required
                    // version
                    &actual_version[..] < min_version
                } else {
                    false
                }
            } else {
                false
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestProps {
    // Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<String>,
    // Extra flags to pass to the compiler
    pub compile_flags: Vec<String>,
    // Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Option<String>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pub pp_exact: Option<PathBuf>,
    // Other crates that should be compiled (typically from the same
    // directory as the test, but for backwards compatibility reasons
    // we also check the auxiliary directory)
    pub aux_builds: Vec<String>,
    // Environment settings to use for compiling
    pub rustc_env: Vec<(String, String)>,
    // Environment settings to use during execution
    pub exec_env: Vec<(String, String)>,
    // Lines to check if they appear in the expected debugger output
    pub check_lines: Vec<String>,
    // Build documentation for all specified aux-builds as well
    pub build_aux_docs: bool,
    // Flag to force a crate to be built with the host architecture
    pub force_host: bool,
    // Check stdout for error-pattern output as well as stderr
    pub check_stdout: bool,
    // Don't force a --crate-type=dylib flag on the command line
    pub no_prefer_dynamic: bool,
    // Run --pretty expanded when running pretty printing tests
    pub pretty_expanded: bool,
    // Which pretty mode are we testing with, default to 'normal'
    pub pretty_mode: String,
    // Only compare pretty output and don't try compiling
    pub pretty_compare_only: bool,
    // Patterns which must not appear in the output of a cfail test.
    pub forbid_output: Vec<String>,
    // Revisions to test for incremental compilation.
    pub revisions: Vec<String>,
    // Directory (if any) to use for incremental compilation.  This is
    // not set by end-users; rather it is set by the incremental
    // testing harness and used when generating compilation
    // arguments. (In particular, it propagates to the aux-builds.)
    pub incremental_dir: Option<PathBuf>,
    // Specifies that a cfail test must actually compile without errors.
    pub must_compile_successfully: bool,
}

impl TestProps {
    pub fn new() -> Self {
        TestProps {
            error_patterns: vec![],
            compile_flags: vec![],
            run_flags: None,
            pp_exact: None,
            aux_builds: vec![],
            revisions: vec![],
            rustc_env: vec![],
            exec_env: vec![],
            check_lines: vec![],
            build_aux_docs: false,
            force_host: false,
            check_stdout: false,
            no_prefer_dynamic: false,
            pretty_expanded: false,
            pretty_mode: format!("normal"),
            pretty_compare_only: false,
            forbid_output: vec![],
            incremental_dir: None,
            must_compile_successfully: false,
        }
    }

    pub fn from_aux_file(&self, testfile: &Path, cfg: Option<&str>) -> Self {
        let mut props = TestProps::new();

        // copy over select properties to the aux build:
        props.incremental_dir = self.incremental_dir.clone();
        props.load_from(testfile, cfg);

        props
    }

    pub fn from_file(testfile: &Path) -> Self {
        let mut props = TestProps::new();
        props.load_from(testfile, None);
        props
    }

    /// Load properties from `testfile` into `props`. If a property is
    /// tied to a particular revision `foo` (indicated by writing
    /// `//[foo]`), then the property is ignored unless `cfg` is
    /// `Some("foo")`.
    pub fn load_from(&mut self, testfile: &Path, cfg: Option<&str>) {
        iter_header(testfile,
                    cfg,
                    &mut |ln| {
            if let Some(ep) = parse_error_pattern(ln) {
                self.error_patterns.push(ep);
            }

            if let Some(flags) = parse_compile_flags(ln) {
                self.compile_flags.extend(flags.split_whitespace()
                    .map(|s| s.to_owned()));
            }

            if let Some(r) = parse_revisions(ln) {
                self.revisions.extend(r);
            }

            if self.run_flags.is_none() {
                self.run_flags = parse_run_flags(ln);
            }

            if self.pp_exact.is_none() {
                self.pp_exact = parse_pp_exact(ln, testfile);
            }

            if !self.build_aux_docs {
                self.build_aux_docs = parse_build_aux_docs(ln);
            }

            if !self.force_host {
                self.force_host = parse_force_host(ln);
            }

            if !self.check_stdout {
                self.check_stdout = parse_check_stdout(ln);
            }

            if !self.no_prefer_dynamic {
                self.no_prefer_dynamic = parse_no_prefer_dynamic(ln);
            }

            if !self.pretty_expanded {
                self.pretty_expanded = parse_pretty_expanded(ln);
            }

            if let Some(m) = parse_pretty_mode(ln) {
                self.pretty_mode = m;
            }

            if !self.pretty_compare_only {
                self.pretty_compare_only = parse_pretty_compare_only(ln);
            }

            if let Some(ab) = parse_aux_build(ln) {
                self.aux_builds.push(ab);
            }

            if let Some(ee) = parse_env(ln, "exec-env") {
                self.exec_env.push(ee);
            }

            if let Some(ee) = parse_env(ln, "rustc-env") {
                self.rustc_env.push(ee);
            }

            if let Some(cl) = parse_check_line(ln) {
                self.check_lines.push(cl);
            }

            if let Some(of) = parse_forbid_output(ln) {
                self.forbid_output.push(of);
            }

            if !self.must_compile_successfully {
                self.must_compile_successfully = parse_must_compile_successfully(ln);
            }
        });

        for key in vec!["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
            match env::var(key) {
                Ok(val) => {
                    if self.exec_env.iter().find(|&&(ref x, _)| *x == key).is_none() {
                        self.exec_env.push((key.to_owned(), val))
                    }
                }
                Err(..) => {}
            }
        }
    }
}

fn iter_header(testfile: &Path, cfg: Option<&str>, it: &mut FnMut(&str)) {
    if testfile.is_dir() {
        return;
    }
    let rdr = BufReader::new(File::open(testfile).unwrap());
    for ln in rdr.lines() {
        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        let ln = ln.unwrap();
        let ln = ln.trim();
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return;
        } else if ln.starts_with("//[") {
            // A comment like `//[foo]` is specific to revision `foo`
            if let Some(close_brace) = ln.find("]") {
                let lncfg = &ln[3..close_brace];
                let matches = match cfg {
                    Some(s) => s == &lncfg[..],
                    None => false,
                };
                if matches {
                    it(&ln[close_brace + 1..]);
                }
            } else {
                panic!("malformed condition directive: expected `//[foo]`, found `{}`",
                       ln)
            }
        } else if ln.starts_with("//") {
            it(&ln[2..]);
        }
    }
    return;
}

fn parse_error_pattern(line: &str) -> Option<String> {
    parse_name_value_directive(line, "error-pattern")
}

fn parse_forbid_output(line: &str) -> Option<String> {
    parse_name_value_directive(line, "forbid-output")
}

fn parse_aux_build(line: &str) -> Option<String> {
    parse_name_value_directive(line, "aux-build")
}

fn parse_compile_flags(line: &str) -> Option<String> {
    parse_name_value_directive(line, "compile-flags")
}

fn parse_revisions(line: &str) -> Option<Vec<String>> {
    parse_name_value_directive(line, "revisions")
        .map(|r| r.split_whitespace().map(|t| t.to_string()).collect())
}

fn parse_run_flags(line: &str) -> Option<String> {
    parse_name_value_directive(line, "run-flags")
}

fn parse_check_line(line: &str) -> Option<String> {
    parse_name_value_directive(line, "check")
}

fn parse_force_host(line: &str) -> bool {
    parse_name_directive(line, "force-host")
}

fn parse_build_aux_docs(line: &str) -> bool {
    parse_name_directive(line, "build-aux-docs")
}

fn parse_check_stdout(line: &str) -> bool {
    parse_name_directive(line, "check-stdout")
}

fn parse_no_prefer_dynamic(line: &str) -> bool {
    parse_name_directive(line, "no-prefer-dynamic")
}

fn parse_pretty_expanded(line: &str) -> bool {
    parse_name_directive(line, "pretty-expanded")
}

fn parse_pretty_mode(line: &str) -> Option<String> {
    parse_name_value_directive(line, "pretty-mode")
}

fn parse_pretty_compare_only(line: &str) -> bool {
    parse_name_directive(line, "pretty-compare-only")
}

fn parse_must_compile_successfully(line: &str) -> bool {
    parse_name_directive(line, "must-compile-successfully")
}

fn parse_env(line: &str, name: &str) -> Option<(String, String)> {
    parse_name_value_directive(line, name).map(|nv| {
        // nv is either FOO or FOO=BAR
        let mut strs: Vec<String> = nv.splitn(2, '=')
            .map(str::to_owned)
            .collect();

        match strs.len() {
            1 => (strs.pop().unwrap(), "".to_owned()),
            2 => {
                let end = strs.pop().unwrap();
                (strs.pop().unwrap(), end)
            }
            n => panic!("Expected 1 or 2 strings, not {}", n),
        }
    })
}

fn parse_pp_exact(line: &str, testfile: &Path) -> Option<PathBuf> {
    if let Some(s) = parse_name_value_directive(line, "pp-exact") {
        Some(PathBuf::from(&s))
    } else {
        if parse_name_directive(line, "pp-exact") {
            testfile.file_name().map(PathBuf::from)
        } else {
            None
        }
    }
}

fn parse_name_directive(line: &str, directive: &str) -> bool {
    // This 'no-' rule is a quick hack to allow pretty-expanded and no-pretty-expanded to coexist
    line.contains(directive) && !line.contains(&("no-".to_owned() + directive))
}

pub fn parse_name_value_directive(line: &str, directive: &str) -> Option<String> {
    let keycolon = format!("{}:", directive);
    if let Some(colon) = line.find(&keycolon) {
        let value = line[(colon + keycolon.len())..line.len()].to_owned();
        debug!("{}: {}", directive, value);
        Some(value)
    } else {
        None
    }
}

pub fn lldb_version_to_int(version_string: &str) -> isize {
    let error_string = format!("Encountered LLDB version string with unexpected format: {}",
                               version_string);
    let error_string = error_string;
    let major: isize = version_string.parse().ok().expect(&error_string);
    return major;
}
