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

#[derive(Clone, Debug)]
pub struct TestProps {
    // Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<String> ,
    // Extra flags to pass to the compiler
    pub compile_flags: Vec<String>,
    // Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Option<String>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pub pp_exact: Option<PathBuf>,
    // Modules from aux directory that should be compiled
    pub aux_builds: Vec<String> ,
    // Environment settings to use for compiling
    pub rustc_env: Vec<(String,String)> ,
    // Environment settings to use during execution
    pub exec_env: Vec<(String,String)> ,
    // Lines to check if they appear in the expected debugger output
    pub check_lines: Vec<String> ,
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
}

// Load any test directives embedded in the file
pub fn load_props(testfile: &Path) -> TestProps {
    let error_patterns = Vec::new();
    let aux_builds = Vec::new();
    let exec_env = Vec::new();
    let run_flags = None;
    let pp_exact = None;
    let check_lines = Vec::new();
    let build_aux_docs = false;
    let force_host = false;
    let check_stdout = false;
    let no_prefer_dynamic = false;
    let pretty_expanded = false;
    let pretty_compare_only = false;
    let forbid_output = Vec::new();
    let mut props = TestProps {
        error_patterns: error_patterns,
        compile_flags: vec![],
        run_flags: run_flags,
        pp_exact: pp_exact,
        aux_builds: aux_builds,
        revisions: vec![],
        rustc_env: vec![],
        exec_env: exec_env,
        check_lines: check_lines,
        build_aux_docs: build_aux_docs,
        force_host: force_host,
        check_stdout: check_stdout,
        no_prefer_dynamic: no_prefer_dynamic,
        pretty_expanded: pretty_expanded,
        pretty_mode: format!("normal"),
        pretty_compare_only: pretty_compare_only,
        forbid_output: forbid_output,
    };
    load_props_into(&mut props, testfile, None);
    props
}

/// Load properties from `testfile` into `props`. If a property is
/// tied to a particular revision `foo` (indicated by writing
/// `//[foo]`), then the property is ignored unless `cfg` is
/// `Some("foo")`.
pub fn load_props_into(props: &mut TestProps, testfile: &Path, cfg: Option<&str>)  {
    iter_header(testfile, cfg, &mut |ln| {
        if let Some(ep) = parse_error_pattern(ln) {
            props.error_patterns.push(ep);
        }

        if let Some(flags) = parse_compile_flags(ln) {
            props.compile_flags.extend(
                flags
                    .split_whitespace()
                    .map(|s| s.to_owned()));
        }

        if let Some(r) = parse_revisions(ln) {
            props.revisions.extend(r);
        }

        if props.run_flags.is_none() {
            props.run_flags = parse_run_flags(ln);
        }

        if props.pp_exact.is_none() {
            props.pp_exact = parse_pp_exact(ln, testfile);
        }

        if !props.build_aux_docs {
            props.build_aux_docs = parse_build_aux_docs(ln);
        }

        if !props.force_host {
            props.force_host = parse_force_host(ln);
        }

        if !props.check_stdout {
            props.check_stdout = parse_check_stdout(ln);
        }

        if !props.no_prefer_dynamic {
            props.no_prefer_dynamic = parse_no_prefer_dynamic(ln);
        }

        if !props.pretty_expanded {
            props.pretty_expanded = parse_pretty_expanded(ln);
        }

        if let Some(m) = parse_pretty_mode(ln) {
            props.pretty_mode = m;
        }

        if !props.pretty_compare_only {
            props.pretty_compare_only = parse_pretty_compare_only(ln);
        }

        if let  Some(ab) = parse_aux_build(ln) {
            props.aux_builds.push(ab);
        }

        if let Some(ee) = parse_env(ln, "exec-env") {
            props.exec_env.push(ee);
        }

        if let Some(ee) = parse_env(ln, "rustc-env") {
            props.rustc_env.push(ee);
        }

        if let Some(cl) =  parse_check_line(ln) {
            props.check_lines.push(cl);
        }

        if let Some(of) = parse_forbid_output(ln) {
            props.forbid_output.push(of);
        }
    });

    for key in vec!["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
        match env::var(key) {
            Ok(val) =>
                if props.exec_env.iter().find(|&&(ref x, _)| *x == key).is_none() {
                    props.exec_env.push((key.to_owned(), val))
                },
            Err(..) => {}
        }
    }
}

pub struct EarlyProps {
    pub ignore: bool,
    pub should_fail: bool,
}

// scan the file to detect whether the test should be ignored and
// whether it should panic; these are two things the test runner needs
// to know early, before actually running the test
pub fn early_props(config: &Config, testfile: &Path) -> EarlyProps {
    let mut props = EarlyProps {
        ignore: false,
        should_fail: false,
    };

    iter_header(testfile, None, &mut |ln| {
        props.ignore =
            props.ignore ||
            parse_name_directive(ln, "ignore-test") ||
            parse_name_directive(ln, &ignore_target(config)) ||
            parse_name_directive(ln, &ignore_architecture(config)) ||
            parse_name_directive(ln, &ignore_stage(config)) ||
            parse_name_directive(ln, &ignore_env(config)) ||
            (config.mode == common::Pretty &&
             parse_name_directive(ln, "ignore-pretty")) ||
            (config.target != config.host &&
             parse_name_directive(ln, "ignore-cross-compile")) ||
            ignore_gdb(config, ln) ||
            ignore_lldb(config, ln);

        props.should_fail =
            props.should_fail ||
            parse_name_directive(ln, "should-fail");
    });

    return props;

    fn ignore_target(config: &Config) -> String {
        format!("ignore-{}", util::get_os(&config.target))
    }
    fn ignore_architecture(config: &Config) -> String {
        format!("ignore-{}", util::get_arch(&config.target))
    }
    fn ignore_stage(config: &Config) -> String {
        format!("ignore-{}",
                config.stage_id.split('-').next().unwrap())
    }
    fn ignore_env(config: &Config) -> String {
        format!("ignore-{}", util::get_env(&config.target).unwrap_or("<unknown>"))
    }
    fn ignore_gdb(config: &Config, line: &str) -> bool {
        if config.mode != common::DebugInfoGdb {
            return false;
        }

        if parse_name_directive(line, "ignore-gdb") {
            return true;
        }

        if let Some(ref actual_version) = config.gdb_version {
            if line.contains("min-gdb-version") {
                let min_version = line.trim()
                                      .split(' ')
                                      .last()
                                      .expect("Malformed GDB version directive");
                // Ignore if actual version is smaller the minimum required
                // version
                gdb_version_to_int(actual_version) <
                    gdb_version_to_int(min_version)
            } else {
                false
            }
        } else {
            false
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
                lldb_version_to_int(actual_version) <
                    lldb_version_to_int(min_version)
            } else {
                false
            }
        } else {
            false
        }
    }
}

fn iter_header(testfile: &Path,
               cfg: Option<&str>,
               it: &mut FnMut(&str)) {
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
                    it(&ln[close_brace+1..]);
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

fn parse_env(line: &str, name: &str) -> Option<(String, String)> {
    parse_name_value_directive(line, name).map(|nv| {
        // nv is either FOO or FOO=BAR
        let mut strs: Vec<String> = nv
                                      .splitn(2, '=')
                                      .map(str::to_owned)
                                      .collect();

        match strs.len() {
          1 => (strs.pop().unwrap(), "".to_owned()),
          2 => {
              let end = strs.pop().unwrap();
              (strs.pop().unwrap(), end)
          }
          n => panic!("Expected 1 or 2 strings, not {}", n)
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

pub fn parse_name_value_directive(line: &str, directive: &str)
                                  -> Option<String> {
    let keycolon = format!("{}:", directive);
    if let Some(colon) = line.find(&keycolon) {
        let value = line[(colon + keycolon.len()) .. line.len()].to_owned();
        debug!("{}: {}", directive, value);
        Some(value)
    } else {
        None
    }
}

pub fn gdb_version_to_int(version_string: &str) -> isize {
    let error_string = format!(
        "Encountered GDB version string with unexpected format: {}",
        version_string);
    let error_string = error_string;

    let components: Vec<&str> = version_string.trim().split('.').collect();

    if components.len() != 2 {
        panic!("{}", error_string);
    }

    let major: isize = components[0].parse().ok().expect(&error_string);
    let minor: isize = components[1].parse().ok().expect(&error_string);

    return major * 1000 + minor;
}

pub fn lldb_version_to_int(version_string: &str) -> isize {
    let error_string = format!(
        "Encountered LLDB version string with unexpected format: {}",
        version_string);
    let error_string = error_string;
    let major: isize = version_string.parse().ok().expect(&error_string);
    return major;
}
