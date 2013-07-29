// Copyright 2012-2013 The Rust Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::mode_run_pass;
use common::mode_run_fail;
use common::mode_compile_fail;
use common::mode_pretty;
use common::config;
use errors;
use header::load_props;
use header::TestProps;
use procsrv;
use util;
use util::logv;

use std::io;
use std::os;
use std::str;
use std::uint;
use std::vec;

use extra::test::MetricMap;

pub fn run(config: config, testfile: ~str) {
    let mut _mm = MetricMap::new();
    run_metrics(config, testfile, &mut _mm);
}

pub fn run_metrics(config: config, testfile: ~str, mm: &mut MetricMap) {
    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        io::stdout().write_str("\n\n");
    }
    let testfile = Path(testfile);
    debug!("running %s", testfile.to_str());
    let props = load_props(&testfile);
    debug!("loaded props");
    match config.mode {
      mode_compile_fail => run_cfail_test(&config, &props, &testfile),
      mode_run_fail => run_rfail_test(&config, &props, &testfile),
      mode_run_pass => run_rpass_test(&config, &props, &testfile),
      mode_pretty => run_pretty_test(&config, &props, &testfile),
      mode_debug_info => run_debuginfo_test(&config, &props, &testfile),
      mode_codegen => run_codegen_test(&config, &props, &testfile, mm)
    }
}

fn run_cfail_test(config: &config, props: &TestProps, testfile: &Path) {
    let ProcRes = compile_test(config, props, testfile);

    if ProcRes.status == 0 {
        fatal_ProcRes(~"compile-fail test compiled successfully!", &ProcRes);
    }

    check_correct_failure_status(&ProcRes);

    let expected_errors = errors::load_errors(testfile);
    if !expected_errors.is_empty() {
        if !props.error_patterns.is_empty() {
            fatal(~"both error pattern and expected errors specified");
        }
        check_expected_errors(expected_errors, testfile, &ProcRes);
    } else {
        check_error_patterns(props, testfile, &ProcRes);
    }
}

fn run_rfail_test(config: &config, props: &TestProps, testfile: &Path) {
    let ProcRes = if !config.jit {
        let ProcRes = compile_test(config, props, testfile);

        if ProcRes.status != 0 {
            fatal_ProcRes(~"compilation failed!", &ProcRes);
        }

        exec_compiled_test(config, props, testfile)
    } else {
        jit_test(config, props, testfile)
    };

    // The value our Makefile configures valgrind to return on failure
    static VALGRIND_ERR: int = 100;
    if ProcRes.status == VALGRIND_ERR {
        fatal_ProcRes(~"run-fail test isn't valgrind-clean!", &ProcRes);
    }

    match config.target {

        ~"arm-linux-androideabi" => {
            if (config.adb_device_status) {
                check_correct_failure_status(&ProcRes);
                check_error_patterns(props, testfile, &ProcRes);
            }
        }

        _=> {
            check_correct_failure_status(&ProcRes);
            check_error_patterns(props, testfile, &ProcRes);
        }
    }
}

fn check_correct_failure_status(ProcRes: &ProcRes) {
    // The value the rust runtime returns on failure
    static RUST_ERR: int = 101;
    if ProcRes.status != RUST_ERR {
        fatal_ProcRes(
            fmt!("failure produced the wrong error code: %d",
                 ProcRes.status),
            ProcRes);
    }
}

fn run_rpass_test(config: &config, props: &TestProps, testfile: &Path) {
    if !config.jit {
        let mut ProcRes = compile_test(config, props, testfile);

        if ProcRes.status != 0 {
            fatal_ProcRes(~"compilation failed!", &ProcRes);
        }

        ProcRes = exec_compiled_test(config, props, testfile);

        if ProcRes.status != 0 {
            fatal_ProcRes(~"test run failed!", &ProcRes);
        }
    } else {
        let ProcRes = jit_test(config, props, testfile);

        if ProcRes.status != 0 { fatal_ProcRes(~"jit failed!", &ProcRes); }
    }
}

fn run_pretty_test(config: &config, props: &TestProps, testfile: &Path) {
    if props.pp_exact.is_some() {
        logv(config, ~"testing for exact pretty-printing");
    } else { logv(config, ~"testing for converging pretty-printing"); }

    let rounds =
        match props.pp_exact { Some(_) => 1, None => 2 };

    let mut srcs = ~[io::read_whole_file_str(testfile).get()];

    let mut round = 0;
    while round < rounds {
        logv(config, fmt!("pretty-printing round %d", round));
        let ProcRes = print_source(config, testfile, srcs[round].clone());

        if ProcRes.status != 0 {
            fatal_ProcRes(fmt!("pretty-printing failed in round %d", round),
                          &ProcRes);
        }

        let ProcRes{ stdout, _ } = ProcRes;
        srcs.push(stdout);
        round += 1;
    }

    let mut expected =
        match props.pp_exact {
          Some(ref file) => {
            let filepath = testfile.dir_path().push_rel(file);
            io::read_whole_file_str(&filepath).get()
          }
          None => { srcs[srcs.len() - 2u].clone() }
        };
    let mut actual = srcs[srcs.len() - 1u].clone();

    if props.pp_exact.is_some() {
        // Now we have to care about line endings
        let cr = ~"\r";
        actual = actual.replace(cr, "");
        expected = expected.replace(cr, "");
    }

    compare_source(expected, actual);

    // Finally, let's make sure it actually appears to remain valid code
    let ProcRes = typecheck_source(config, props, testfile, actual);

    if ProcRes.status != 0 {
        fatal_ProcRes(~"pretty-printed source does not typecheck", &ProcRes);
    }

    return;

    fn print_source(config: &config, testfile: &Path, src: ~str) -> ProcRes {
        compose_and_run(config, testfile, make_pp_args(config, testfile),
                        ~[], config.compile_lib_path, Some(src))
    }

    fn make_pp_args(config: &config, _testfile: &Path) -> ProcArgs {
        let args = ~[~"-", ~"--pretty", ~"normal"];
        return ProcArgs {prog: config.rustc_path.to_str(), args: args};
    }

    fn compare_source(expected: &str, actual: &str) {
        if expected != actual {
            error(~"pretty-printed source does not match expected source");
            let msg =
                fmt!("\n\
expected:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
actual:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
\n",
                     expected, actual);
            io::stdout().write_str(msg);
            fail!();
        }
    }

    fn typecheck_source(config: &config, props: &TestProps,
                        testfile: &Path, src: ~str) -> ProcRes {
        let args = make_typecheck_args(config, props, testfile);
        compose_and_run_compiler(config, props, testfile, args, Some(src))
    }

    fn make_typecheck_args(config: &config, props: &TestProps, testfile: &Path) -> ProcArgs {
        let mut args = ~[~"-",
                         ~"--no-trans", ~"--lib",
                         ~"-L", config.build_base.to_str(),
                         ~"-L",
                         aux_output_dir_name(config, testfile).to_str()];
        args.push_all_move(split_maybe_args(&config.rustcflags));
        args.push_all_move(split_maybe_args(&props.compile_flags));
        return ProcArgs {prog: config.rustc_path.to_str(), args: args};
    }
}

fn run_debuginfo_test(config: &config, props: &TestProps, testfile: &Path) {
    // do not optimize debuginfo tests
    let mut config = match config.rustcflags {
        Some(ref flags) => config {
            rustcflags: Some(flags.replace("-O", "")),
            .. (*config).clone()
        },
        None => (*config).clone()
    };
    let config = &mut config;
    let cmds = props.debugger_cmds.connect("\n");
    let check_lines = props.check_lines.clone();

    // compile test file (it shoud have 'compile-flags:-g' in the header)
    let mut ProcRes = compile_test(config, props, testfile);
    if ProcRes.status != 0 {
        fatal_ProcRes(~"compilation failed!", &ProcRes);
    }

    // write debugger script
    let script_str = cmds.append("\nquit\n");
    debug!("script_str = %s", script_str);
    dump_output_file(config, testfile, script_str, "debugger.script");

    // run debugger script with gdb
    #[cfg(windows)]
    fn debugger() -> ~str { ~"gdb.exe" }
    #[cfg(unix)]
    fn debugger() -> ~str { ~"gdb" }
    let debugger_script = make_out_name(config, testfile, "debugger.script");
    let debugger_opts = ~[~"-quiet", ~"-batch", ~"-nx",
                          ~"-command=" + debugger_script.to_str(),
                          make_exe_name(config, testfile).to_str()];
    let ProcArgs = ProcArgs {prog: debugger(), args: debugger_opts};
    ProcRes = compose_and_run(config, testfile, ProcArgs, ~[], "", None);
    if ProcRes.status != 0 {
        fatal(~"gdb failed to execute");
    }

    let num_check_lines = check_lines.len();
    if num_check_lines > 0 {
        // check if each line in props.check_lines appears in the
        // output (in order)
        let mut i = 0u;
        for ProcRes.stdout.line_iter().advance |line| {
            if check_lines[i].trim() == line.trim() {
                i += 1u;
            }
            if i == num_check_lines {
                // all lines checked
                break;
            }
        }
        if i != num_check_lines {
            fatal_ProcRes(fmt!("line not found in debugger output: %s",
                               check_lines[i]), &ProcRes);
        }
    }
}

fn check_error_patterns(props: &TestProps,
                        testfile: &Path,
                        ProcRes: &ProcRes) {
    if props.error_patterns.is_empty() {
        fatal(~"no error pattern specified in " + testfile.to_str());
    }

    if ProcRes.status == 0 {
        fatal(~"process did not return an error status");
    }

    let mut next_err_idx = 0u;
    let mut next_err_pat = &props.error_patterns[next_err_idx];
    let mut done = false;
    for ProcRes.stderr.line_iter().advance |line| {
        if line.contains(*next_err_pat) {
            debug!("found error pattern %s", *next_err_pat);
            next_err_idx += 1u;
            if next_err_idx == props.error_patterns.len() {
                debug!("found all error patterns");
                done = true;
                break;
            }
            next_err_pat = &props.error_patterns[next_err_idx];
        }
    }
    if done { return; }

    let missing_patterns =
        props.error_patterns.slice(next_err_idx, props.error_patterns.len());
    if missing_patterns.len() == 1u {
        fatal_ProcRes(fmt!("error pattern '%s' not found!",
                           missing_patterns[0]), ProcRes);
    } else {
        for missing_patterns.iter().advance |pattern| {
            error(fmt!("error pattern '%s' not found!", *pattern));
        }
        fatal_ProcRes(~"multiple error patterns not found", ProcRes);
    }
}

fn check_expected_errors(expected_errors: ~[errors::ExpectedError],
                         testfile: &Path,
                         ProcRes: &ProcRes) {

    // true if we found the error in question
    let mut found_flags = vec::from_elem(
        expected_errors.len(), false);

    if ProcRes.status == 0 {
        fatal(~"process did not return an error status");
    }

    let prefixes = expected_errors.iter().transform(|ee| {
        fmt!("%s:%u:", testfile.to_str(), ee.line)
    }).collect::<~[~str]>();

    fn to_lower( s : &str ) -> ~str {
        let i = s.iter();
        let c : ~[char] = i.transform( |c| {
            if c.is_ascii() {
                c.to_ascii().to_lower().to_char()
            } else {
                c
            }
        } ).collect();
        str::from_chars( c )
    }

    #[cfg(target_os = "win32")]
    fn prefix_matches( line : &str, prefix : &str ) -> bool {
        to_lower(line).starts_with( to_lower(prefix) )
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn prefix_matches( line : &str, prefix : &str ) -> bool {
        line.starts_with( prefix )
    }

    // Scan and extract our error/warning messages,
    // which look like:
    //    filename:line1:col1: line2:col2: *error:* msg
    //    filename:line1:col1: line2:col2: *warning:* msg
    // where line1:col1: is the starting point, line2:col2:
    // is the ending point, and * represents ANSI color codes.
    for ProcRes.stderr.line_iter().advance |line| {
        let mut was_expected = false;
        for expected_errors.iter().enumerate().advance |(i, ee)| {
            if !found_flags[i] {
                debug!("prefix=%s ee.kind=%s ee.msg=%s line=%s",
                       prefixes[i], ee.kind, ee.msg, line);
                if (prefix_matches(line, prefixes[i]) &&
                    line.contains(ee.kind) &&
                    line.contains(ee.msg)) {
                    found_flags[i] = true;
                    was_expected = true;
                    break;
                }
            }
        }

        // ignore this msg which gets printed at the end
        if line.contains("aborting due to") {
            was_expected = true;
        }

        if !was_expected && is_compiler_error_or_warning(line) {
            fatal_ProcRes(fmt!("unexpected compiler error or warning: '%s'",
                               line),
                          ProcRes);
        }
    }

    for uint::range(0u, found_flags.len()) |i| {
        if !found_flags[i] {
            let ee = &expected_errors[i];
            fatal_ProcRes(fmt!("expected %s on line %u not found: %s",
                               ee.kind, ee.line, ee.msg), ProcRes);
        }
    }
}

fn is_compiler_error_or_warning(line: &str) -> bool {
    let mut i = 0u;
    return
        scan_until_char(line, ':', &mut i) &&
        scan_char(line, ':', &mut i) &&
        scan_integer(line, &mut i) &&
        scan_char(line, ':', &mut i) &&
        scan_integer(line, &mut i) &&
        scan_char(line, ':', &mut i) &&
        scan_char(line, ' ', &mut i) &&
        scan_integer(line, &mut i) &&
        scan_char(line, ':', &mut i) &&
        scan_integer(line, &mut i) &&
        scan_char(line, ' ', &mut i) &&
        (scan_string(line, "error", &mut i) ||
         scan_string(line, "warning", &mut i));
}

fn scan_until_char(haystack: &str, needle: char, idx: &mut uint) -> bool {
    if *idx >= haystack.len() {
        return false;
    }
    let opt = haystack.slice_from(*idx).find(needle);
    if opt.is_none() {
        return false;
    }
    *idx = opt.get();
    return true;
}

fn scan_char(haystack: &str, needle: char, idx: &mut uint) -> bool {
    if *idx >= haystack.len() {
        return false;
    }
    let range = haystack.char_range_at(*idx);
    if range.ch != needle {
        return false;
    }
    *idx = range.next;
    return true;
}

fn scan_integer(haystack: &str, idx: &mut uint) -> bool {
    let mut i = *idx;
    while i < haystack.len() {
        let range = haystack.char_range_at(i);
        if range.ch < '0' || '9' < range.ch {
            break;
        }
        i = range.next;
    }
    if i == *idx {
        return false;
    }
    *idx = i;
    return true;
}

fn scan_string(haystack: &str, needle: &str, idx: &mut uint) -> bool {
    let mut haystack_i = *idx;
    let mut needle_i = 0u;
    while needle_i < needle.len() {
        if haystack_i >= haystack.len() {
            return false;
        }
        let range = haystack.char_range_at(haystack_i);
        haystack_i = range.next;
        if !scan_char(needle, range.ch, &mut needle_i) {
            return false;
        }
    }
    *idx = haystack_i;
    return true;
}

struct ProcArgs {prog: ~str, args: ~[~str]}

struct ProcRes {status: int, stdout: ~str, stderr: ~str, cmdline: ~str}

fn compile_test(config: &config, props: &TestProps,
                testfile: &Path) -> ProcRes {
    compile_test_(config, props, testfile, [])
}

fn jit_test(config: &config, props: &TestProps, testfile: &Path) -> ProcRes {
    compile_test_(config, props, testfile, [~"--jit"])
}

fn compile_test_(config: &config, props: &TestProps,
                 testfile: &Path, extra_args: &[~str]) -> ProcRes {
    let link_args = ~[~"-L", aux_output_dir_name(config, testfile).to_str()];
    let args = make_compile_args(config, props, link_args + extra_args,
                                 make_exe_name, testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn exec_compiled_test(config: &config, props: &TestProps,
                      testfile: &Path) -> ProcRes {

    // If testing the new runtime then set the RUST_NEWRT env var
    let env = props.exec_env.clone();
    let env = if config.newrt { env + &[(~"RUST_NEWRT", ~"1")] } else { env };

    match config.target {

        ~"arm-linux-androideabi" => {
            if (config.adb_device_status) {
                _arm_exec_compiled_test(config, props, testfile)
            } else {
                _dummy_exec_compiled_test(config, props, testfile)
            }
        }

        _=> {
            compose_and_run(config, testfile,
                            make_run_args(config, props, testfile),
                            env,
                            config.run_lib_path, None)
        }
    }
}

fn compose_and_run_compiler(
    config: &config,
    props: &TestProps,
    testfile: &Path,
    args: ProcArgs,
    input: Option<~str>) -> ProcRes {

    if !props.aux_builds.is_empty() {
        ensure_dir(&aux_output_dir_name(config, testfile));
    }

    let extra_link_args = ~[~"-L",
                            aux_output_dir_name(config, testfile).to_str()];

    for props.aux_builds.iter().advance |rel_ab| {
        let abs_ab = config.aux_base.push_rel(&Path(*rel_ab));
        let aux_args =
            make_compile_args(config, props, ~[~"--lib"] + extra_link_args,
                              |a,b| make_lib_name(a, b, testfile), &abs_ab);
        let auxres = compose_and_run(config, &abs_ab, aux_args, ~[],
                                     config.compile_lib_path, None);
        if auxres.status != 0 {
            fatal_ProcRes(
                fmt!("auxiliary build of %s failed to compile: ",
                     abs_ab.to_str()),
                &auxres);
        }

        match config.target {

            ~"arm-linux-androideabi" => {
                if (config.adb_device_status) {
                    _arm_push_aux_shared_library(config, testfile);
                }
            }

            _=> { }
        }
    }

    compose_and_run(config, testfile, args, ~[],
                    config.compile_lib_path, input)
}

fn ensure_dir(path: &Path) {
    if os::path_is_dir(path) { return; }
    if !os::make_dir(path, 0x1c0i32) {
        fail!("can't make dir %s", path.to_str());
    }
}

fn compose_and_run(config: &config, testfile: &Path,
                   ProcArgs{ args, prog }: ProcArgs,
                   procenv: ~[(~str, ~str)],
                   lib_path: &str,
                   input: Option<~str>) -> ProcRes {
    return program_output(config, testfile, lib_path,
                          prog, args, procenv, input);
}

fn make_compile_args(config: &config, props: &TestProps, extras: ~[~str],
                     xform: &fn(&config, (&Path)) -> Path,
                     testfile: &Path) -> ProcArgs {
    let mut args = ~[testfile.to_str(),
                     ~"-o", xform(config, testfile).to_str(),
                     ~"-L", config.build_base.to_str()]
        + extras;
    args.push_all_move(split_maybe_args(&config.rustcflags));
    args.push_all_move(split_maybe_args(&props.compile_flags));
    return ProcArgs {prog: config.rustc_path.to_str(), args: args};
}

fn make_lib_name(config: &config, auxfile: &Path, testfile: &Path) -> Path {
    // what we return here is not particularly important, as it
    // happens; rustc ignores everything except for the directory.
    let auxname = output_testname(auxfile);
    aux_output_dir_name(config, testfile).push_rel(&auxname)
}

fn make_exe_name(config: &config, testfile: &Path) -> Path {
    Path(output_base_name(config, testfile).to_str() + os::EXE_SUFFIX)
}

fn make_run_args(config: &config, _props: &TestProps, testfile: &Path) ->
   ProcArgs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    let mut args = split_maybe_args(&config.runtool);
    args.push(make_exe_name(config, testfile).to_str());
    let prog = args.shift();
    return ProcArgs {prog: prog, args: args};
}

fn split_maybe_args(argstr: &Option<~str>) -> ~[~str] {
    match *argstr {
        Some(ref s) => {
            s.split_iter(' ')
                .filter_map(|s| if s.is_whitespace() {None} else {Some(s.to_owned())})
                .collect()
        }
        None => ~[]
    }
}

fn program_output(config: &config, testfile: &Path, lib_path: &str, prog: ~str,
                  args: ~[~str], env: ~[(~str, ~str)],
                  input: Option<~str>) -> ProcRes {
    let cmdline =
        {
            let cmdline = make_cmdline(lib_path, prog, args);
            logv(config, fmt!("executing %s", cmdline));
            cmdline
        };
    let procsrv::Result{ out, err, status } =
            procsrv::run(lib_path, prog, args, env, input);
    dump_output(config, testfile, out, err);
    return ProcRes {status: status,
         stdout: out,
         stderr: err,
         cmdline: cmdline};
}

// Linux and mac don't require adjusting the library search path
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn make_cmdline(_libpath: &str, prog: &str, args: &[~str]) -> ~str {
    fmt!("%s %s", prog, args.connect(" "))
}

#[cfg(target_os = "win32")]
fn make_cmdline(libpath: &str, prog: &str, args: &[~str]) -> ~str {
    fmt!("%s %s %s", lib_path_cmd_prefix(libpath), prog,
         args.connect(" "))
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
fn lib_path_cmd_prefix(path: &str) -> ~str {
    fmt!("%s=\"%s\"", util::lib_path_env_var(), util::make_new_path(path))
}

fn dump_output(config: &config, testfile: &Path, out: &str, err: &str) {
    dump_output_file(config, testfile, out, "out");
    dump_output_file(config, testfile, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

fn dump_output_file(config: &config, testfile: &Path,
                    out: &str, extension: &str) {
    let outfile = make_out_name(config, testfile, extension);
    let writer =
        io::file_writer(&outfile, [io::Create, io::Truncate]).unwrap();
    writer.write_str(out);
}

fn make_out_name(config: &config, testfile: &Path, extension: &str) -> Path {
    output_base_name(config, testfile).with_filetype(extension)
}

fn aux_output_dir_name(config: &config, testfile: &Path) -> Path {
    Path(output_base_name(config, testfile).to_str() + ".libaux")
}

fn output_testname(testfile: &Path) -> Path {
    Path(testfile.filestem().get())
}

fn output_base_name(config: &config, testfile: &Path) -> Path {
    config.build_base
        .push_rel(&output_testname(testfile))
        .with_filetype(config.stage_id)
}

fn maybe_dump_to_stdout(config: &config, out: &str, err: &str) {
    if config.verbose {
        let sep1 = fmt!("------%s------------------------------", "stdout");
        let sep2 = fmt!("------%s------------------------------", "stderr");
        let sep3 = ~"------------------------------------------";
        io::stdout().write_line(sep1);
        io::stdout().write_line(out);
        io::stdout().write_line(sep2);
        io::stdout().write_line(err);
        io::stdout().write_line(sep3);
    }
}

fn error(err: ~str) { io::stdout().write_line(fmt!("\nerror: %s", err)); }

fn fatal(err: ~str) -> ! { error(err); fail!(); }

fn fatal_ProcRes(err: ~str, ProcRes: &ProcRes) -> ! {
    let msg =
        fmt!("\n\
error: %s\n\
command: %s\n\
stdout:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
stderr:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
\n",
             err, ProcRes.cmdline, ProcRes.stdout, ProcRes.stderr);
    io::stdout().write_str(msg);
    fail!();
}

fn _arm_exec_compiled_test(config: &config, props: &TestProps,
                      testfile: &Path) -> ProcRes {

    let args = make_run_args(config, props, testfile);
    let cmdline = make_cmdline("", args.prog, args.args);

    // get bare program string
    let mut tvec: ~[~str] = args.prog.split_iter('/').transform(|ts| ts.to_owned()).collect();
    let prog_short = tvec.pop();

    // copy to target
    let copy_result = procsrv::run("", config.adb_path,
        [~"push", args.prog.clone(), config.adb_test_dir.clone()],
        ~[(~"",~"")], Some(~""));

    if config.verbose {
        io::stdout().write_str(fmt!("push (%s) %s %s %s",
            config.target, args.prog,
            copy_result.out, copy_result.err));
    }

    logv(config, fmt!("executing (%s) %s", config.target, cmdline));

    let mut runargs = ~[];

    // run test via adb_run_wrapper
    runargs.push(~"shell");
    runargs.push(fmt!("%s/adb_run_wrapper.sh", config.adb_test_dir));
    runargs.push(fmt!("%s", config.adb_test_dir));
    runargs.push(fmt!("%s", prog_short));

    for args.args.iter().advance |tv| {
        runargs.push(tv.to_owned());
    }

    procsrv::run("", config.adb_path, runargs, ~[(~"",~"")], Some(~""));

    // get exitcode of result
    runargs = ~[];
    runargs.push(~"shell");
    runargs.push(~"cat");
    runargs.push(fmt!("%s/%s.exitcode", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: exitcode_out, err: _, status: _ } =
        procsrv::run("", config.adb_path, runargs, ~[(~"",~"")],
                     Some(~""));

    let mut exitcode : int = 0;
    for exitcode_out.iter().advance |c| {
        if !c.is_digit() { break; }
        exitcode = exitcode * 10 + match c {
            '0' .. '9' => c as int - ('0' as int),
            _ => 101,
        }
    }

    // get stdout of result
    runargs = ~[];
    runargs.push(~"shell");
    runargs.push(~"cat");
    runargs.push(fmt!("%s/%s.stdout", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stdout_out, err: _, status: _ } =
        procsrv::run("", config.adb_path, runargs, ~[(~"",~"")], Some(~""));

    // get stderr of result
    runargs = ~[];
    runargs.push(~"shell");
    runargs.push(~"cat");
    runargs.push(fmt!("%s/%s.stderr", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stderr_out, err: _, status: _ } =
        procsrv::run("", config.adb_path, runargs, ~[(~"",~"")], Some(~""));

    dump_output(config, testfile, stdout_out, stderr_out);

    ProcRes {status: exitcode, stdout: stdout_out, stderr: stderr_out, cmdline: cmdline }
}

fn _dummy_exec_compiled_test(config: &config, props: &TestProps,
                      testfile: &Path) -> ProcRes {

    let args = make_run_args(config, props, testfile);
    let cmdline = make_cmdline("", args.prog, args.args);

    match config.mode {
        mode_run_fail => ProcRes {status: 101, stdout: ~"",
                                 stderr: ~"", cmdline: cmdline},
        _             => ProcRes {status: 0, stdout: ~"",
                                 stderr: ~"", cmdline: cmdline}
    }
}

fn _arm_push_aux_shared_library(config: &config, testfile: &Path) {
    let tstr = aux_output_dir_name(config, testfile).to_str();

    let dirs = os::list_dir_path(&Path(tstr));
    for dirs.iter().advance |file| {

        if (file.filetype() == Some(~".so")) {

            let copy_result = procsrv::run("", config.adb_path,
                [~"push", file.to_str(), config.adb_test_dir.clone()],
                ~[(~"",~"")], Some(~""));

            if config.verbose {
                io::stdout().write_str(fmt!("push (%s) %s %s %s",
                    config.target, file.to_str(),
                    copy_result.out, copy_result.err));
            }
        }
    }
}

// codegen tests (vs. clang)

fn make_o_name(config: &config, testfile: &Path) -> Path {
    output_base_name(config, testfile).with_filetype("o")
}

fn append_suffix_to_stem(p: &Path, suffix: &str) -> Path {
    if suffix.len() == 0 {
        (*p).clone()
    } else {
        let stem = p.filestem().get();
        p.with_filestem(stem + "-" + suffix)
    }
}

fn compile_test_and_save_bitcode(config: &config, props: &TestProps,
                                 testfile: &Path) -> ProcRes {
    let link_args = ~[~"-L", aux_output_dir_name(config, testfile).to_str()];
    let llvm_args = ~[~"-c", ~"--lib", ~"--save-temps"];
    let args = make_compile_args(config, props,
                                 link_args + llvm_args,
                                 make_o_name, testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn compile_cc_with_clang_and_save_bitcode(config: &config, _props: &TestProps,
                                          testfile: &Path) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_filetype("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, "clang");
    let ProcArgs = ProcArgs {
        prog: config.clang_path.get_ref().to_str(),
        args: ~[~"-c",
                ~"-emit-llvm",
                ~"-o", bitcodefile.to_str(),
                testfile.with_filetype("cc").to_str() ]
    };
    compose_and_run(config, testfile, ProcArgs, ~[], "", None)
}

fn extract_function_from_bitcode(config: &config, _props: &TestProps,
                                 fname: &str, testfile: &Path,
                                 suffix: &str) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_filetype("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, suffix);
    let extracted_bc = append_suffix_to_stem(&bitcodefile, "extract");
    let ProcArgs = ProcArgs {
        prog: config.llvm_bin_path.get_ref().push("llvm-extract").to_str(),
        args: ~[~"-func=" + fname,
                ~"-o=" + extracted_bc.to_str(),
                bitcodefile.to_str() ]
    };
    compose_and_run(config, testfile, ProcArgs, ~[], "", None)
}

fn disassemble_extract(config: &config, _props: &TestProps,
                       testfile: &Path, suffix: &str) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_filetype("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, suffix);
    let extracted_bc = append_suffix_to_stem(&bitcodefile, "extract");
    let extracted_ll = extracted_bc.with_filetype("ll");
    let ProcArgs = ProcArgs {
        prog: config.llvm_bin_path.get_ref().push("llvm-dis").to_str(),
        args: ~[~"-o=" + extracted_ll.to_str(),
                extracted_bc.to_str() ]
    };
    compose_and_run(config, testfile, ProcArgs, ~[], "", None)
}


fn count_extracted_lines(p: &Path) -> uint {
    let x = io::read_whole_file_str(&p.with_filetype("ll")).get();
    x.line_iter().len_()
}


fn run_codegen_test(config: &config, props: &TestProps,
                    testfile: &Path, mm: &mut MetricMap) {

    if config.llvm_bin_path.is_none() {
        fatal(~"missing --llvm-bin-path");
    }

    if config.clang_path.is_none() {
        fatal(~"missing --clang-path");
    }

    let mut ProcRes = compile_test_and_save_bitcode(config, props, testfile);
    if ProcRes.status != 0 {
        fatal_ProcRes(~"compilation failed!", &ProcRes);
    }

    ProcRes = extract_function_from_bitcode(config, props, "test", testfile, "");
    if ProcRes.status != 0 {
        fatal_ProcRes(~"extracting 'test' function failed", &ProcRes);
    }

    ProcRes = disassemble_extract(config, props, testfile, "");
    if ProcRes.status != 0 {
        fatal_ProcRes(~"disassembling extract failed", &ProcRes);
    }


    let mut ProcRes = compile_cc_with_clang_and_save_bitcode(config, props, testfile);
    if ProcRes.status != 0 {
        fatal_ProcRes(~"compilation failed!", &ProcRes);
    }

    ProcRes = extract_function_from_bitcode(config, props, "test", testfile, "clang");
    if ProcRes.status != 0 {
        fatal_ProcRes(~"extracting 'test' function failed", &ProcRes);
    }

    ProcRes = disassemble_extract(config, props, testfile, "clang");
    if ProcRes.status != 0 {
        fatal_ProcRes(~"disassembling extract failed", &ProcRes);
    }

    let base = output_base_name(config, testfile);
    let base_extract = append_suffix_to_stem(&base, "extract");

    let base_clang = append_suffix_to_stem(&base, "clang");
    let base_clang_extract = append_suffix_to_stem(&base_clang, "extract");

    let base_lines = count_extracted_lines(&base_extract);
    let clang_lines = count_extracted_lines(&base_clang_extract);

    mm.insert_metric("clang-codegen-ratio",
                     (base_lines as f64) / (clang_lines as f64),
                     0.001);
}

