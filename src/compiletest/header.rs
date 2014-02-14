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
    error_patterns: ~[~str],
    // Extra flags to pass to the compiler
    compile_flags: Option<~str>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pp_exact: Option<Path>,
    // Modules from aux directory that should be compiled
    aux_builds: ~[~str],
    // Environment settings to use during execution
    exec_env: ~[(~str,~str)],
    // Commands to be given to the debugger, when testing debug info
    debugger_cmds: ~[~str],
    // Lines to check if they appear in the expected debugger output
    check_lines: ~[~str],
    // Flag to force a crate to be built with the host architecture
    force_host: bool,
    // Check stdout for error-pattern output as well as stderr
    check_stdout: bool,
}

// Load any test directives embedded in the file
pub fn load_props(testfile: &Path) -> TestProps {
    let mut error_patterns = ~[];
    let mut aux_builds = ~[];
    let mut exec_env = ~[];
    let mut compile_flags = None;
    let mut pp_exact = None;
    let mut debugger_cmds = ~[];
    let mut check_lines = ~[];
    let mut force_host = false;
    let mut check_stdout = false;
    iter_header(testfile, |ln| {
        match parse_error_pattern(ln) {
          Some(ep) => error_patterns.push(ep),
          None => ()
        };

        if compile_flags.is_none() {
            compile_flags = parse_compile_flags(ln);
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
    return TestProps {
        error_patterns: error_patterns,
        compile_flags: compile_flags,
        pp_exact: pp_exact,
        aux_builds: aux_builds,
        exec_env: exec_env,
        debugger_cmds: debugger_cmds,
        check_lines: check_lines,
        force_host: force_host,
        check_stdout: check_stdout,
    };
}

pub fn is_test_ignored(config: &config, testfile: &Path) -> bool {
    fn ignore_target(config: &config) -> ~str {
        ~"ignore-" + util::get_os(config.target)
    }
    fn ignore_stage(config: &config) -> ~str {
        ~"ignore-" + config.stage_id.split('-').next().unwrap()
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
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return true;
        } else { if !(it(ln.trim())) { return false; } }
    }
    return true;
}

fn parse_error_pattern(line: &str) -> Option<~str> {
    parse_name_value_directive(line, ~"error-pattern")
}

fn parse_aux_build(line: &str) -> Option<~str> {
    parse_name_value_directive(line, ~"aux-build")
}

fn parse_compile_flags(line: &str) -> Option<~str> {
    parse_name_value_directive(line, ~"compile-flags")
}

fn parse_debugger_cmd(line: &str) -> Option<~str> {
    parse_name_value_directive(line, ~"debugger")
}

fn parse_check_line(line: &str) -> Option<~str> {
    parse_name_value_directive(line, ~"check")
}

fn parse_force_host(line: &str) -> bool {
    parse_name_directive(line, "force-host")
}

fn parse_check_stdout(line: &str) -> bool {
    parse_name_directive(line, "check-stdout")
}

fn parse_exec_env(line: &str) -> Option<(~str, ~str)> {
    parse_name_value_directive(line, ~"exec-env").map(|nv| {
        // nv is either FOO or FOO=BAR
        let mut strs: ~[~str] = nv.splitn('=', 1).map(|s| s.to_owned()).collect();

        match strs.len() {
          1u => (strs.pop().unwrap(), ~""),
          2u => {
              let end = strs.pop().unwrap();
              (strs.pop().unwrap(), end)
          }
          n => fail!("Expected 1 or 2 strings, not {}", n)
        }
    })
}

fn parse_pp_exact(line: &str, testfile: &Path) -> Option<Path> {
    match parse_name_value_directive(line, ~"pp-exact") {
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
