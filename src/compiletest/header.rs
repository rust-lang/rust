// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::config;
use common;
use util;

pub struct TestProps {
    // Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<~str> ,
    // Extra flags to pass to the compiler
    pub compile_flags: Option<~str>,
    // Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Option<~str>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pub pp_exact: Option<Path>,
    // Modules from aux directory that should be compiled
    pub aux_builds: Vec<~str> ,
    // Environment settings to use during execution
    pub exec_env: Vec<(~str,~str)> ,
    // Commands to be given to the debugger, when testing debug info
    pub debugger_cmds: Vec<~str> ,
    // Lines to check if they appear in the expected debugger output
    pub check_lines: Vec<~str> ,
    // Flag to force a crate to be built with the host architecture
    pub force_host: bool,
    // Check stdout for error-pattern output as well as stderr
    pub check_stdout: bool,
    // Don't force a --crate-type=dylib flag on the command line
    pub no_prefer_dynamic: bool,
}

// Load any test directives embedded in the file
pub fn load_props(testfile: &Path) -> TestProps {
    let mut error_patterns = Vec::new();
    let mut aux_builds = Vec::new();
    let mut exec_env = Vec::new();
    let mut compile_flags = None;
    let mut run_flags = None;
    let mut pp_exact = None;
    let mut debugger_cmds = Vec::new();
    let mut check_lines = Vec::new();
    let mut force_host = false;
    let mut check_stdout = false;
    let mut no_prefer_dynamic = false;
    iter_header(testfile, |ln| {
        match parse_error_pattern(ln) {
          Some(ep) => error_patterns.push(ep),
          None => ()
        };

        if compile_flags.is_none() {
            compile_flags = parse_compile_flags(ln);
        }

        if run_flags.is_none() {
            run_flags = parse_run_flags(ln);
        }

        if pp_exact.is_none() {
            pp_exact = parse_pp_exact(ln, testfile);
        }

        if !force_host {
            force_host = parse_force_host(ln);
        }

        if !check_stdout {
            check_stdout = parse_check_stdout(ln);
        }

        if !no_prefer_dynamic {
            no_prefer_dynamic = parse_no_prefer_dynamic(ln);
        }

        match parse_aux_build(ln) {
            Some(ab) => { aux_builds.push(ab); }
            None => {}
        }

        match parse_exec_env(ln) {
            Some(ee) => { exec_env.push(ee); }
            None => {}
        }

        match parse_debugger_cmd(ln) {
            Some(dc) => debugger_cmds.push(dc),
            None => ()
        };

        match parse_check_line(ln) {
            Some(cl) => check_lines.push(cl),
            None => ()
        };

        true
    });

    TestProps {
        error_patterns: error_patterns,
        compile_flags: compile_flags,
        run_flags: run_flags,
        pp_exact: pp_exact,
        aux_builds: aux_builds,
        exec_env: exec_env,
        debugger_cmds: debugger_cmds,
        check_lines: check_lines,
        force_host: force_host,
        check_stdout: check_stdout,
        no_prefer_dynamic: no_prefer_dynamic,
    }
}

pub fn is_test_ignored(config: &config, testfile: &Path) -> bool {
    fn ignore_target(config: &config) -> ~str {
        "ignore-".to_owned() + util::get_os(config.target)
    }
    fn ignore_stage(config: &config) -> ~str {
        "ignore-".to_owned() + config.stage_id.split('-').next().unwrap()
    }

    let val = iter_header(testfile, |ln| {
        if parse_name_directive(ln, "ignore-test") { false }
        else if parse_name_directive(ln, ignore_target(config)) { false }
        else if parse_name_directive(ln, ignore_stage(config)) { false }
        else if config.mode == common::mode_pretty &&
            parse_name_directive(ln, "ignore-pretty") { false }
        else if config.target != config.host &&
            parse_name_directive(ln, "ignore-cross-compile") { false }
        else { true }
    });

    !val
}

fn iter_header(testfile: &Path, it: |&str| -> bool) -> bool {
    use std::io::{BufferedReader, File};

    let mut rdr = BufferedReader::new(File::open(testfile).unwrap());
    for ln in rdr.lines() {
        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        let ln = ln.unwrap();
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return true;
        } else { if !(it(ln.trim())) { return false; } }
    }
    return true;
}

fn parse_error_pattern(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "error-pattern".to_owned())
}

fn parse_aux_build(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "aux-build".to_owned())
}

fn parse_compile_flags(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "compile-flags".to_owned())
}

fn parse_run_flags(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "run-flags".to_owned())
}

fn parse_debugger_cmd(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "debugger".to_owned())
}

fn parse_check_line(line: &str) -> Option<~str> {
    parse_name_value_directive(line, "check".to_owned())
}

fn parse_force_host(line: &str) -> bool {
    parse_name_directive(line, "force-host")
}

fn parse_check_stdout(line: &str) -> bool {
    parse_name_directive(line, "check-stdout")
}

fn parse_no_prefer_dynamic(line: &str) -> bool {
    parse_name_directive(line, "no-prefer-dynamic")
}

fn parse_exec_env(line: &str) -> Option<(~str, ~str)> {
    parse_name_value_directive(line, "exec-env".to_owned()).map(|nv| {
        // nv is either FOO or FOO=BAR
        let mut strs: Vec<~str> = nv.splitn('=', 1).map(|s| s.to_owned()).collect();

        match strs.len() {
          1u => (strs.pop().unwrap(), "".to_owned()),
          2u => {
              let end = strs.pop().unwrap();
              (strs.pop().unwrap(), end)
          }
          n => fail!("Expected 1 or 2 strings, not {}", n)
        }
    })
}

fn parse_pp_exact(line: &str, testfile: &Path) -> Option<Path> {
    match parse_name_value_directive(line, "pp-exact".to_owned()) {
      Some(s) => Some(Path::new(s)),
      None => {
        if parse_name_directive(line, "pp-exact") {
            testfile.filename().map(|s| Path::new(s))
        } else {
            None
        }
      }
    }
}

fn parse_name_directive(line: &str, directive: &str) -> bool {
    line.contains(directive)
}

fn parse_name_value_directive(line: &str,
                              directive: ~str) -> Option<~str> {
    let keycolon = directive + ":";
    match line.find_str(keycolon) {
        Some(colon) => {
            let value = line.slice(colon + keycolon.len(),
                                   line.len()).to_owned();
            debug!("{}: {}", directive,  value);
            Some(value)
        }
        None => None
    }
}
