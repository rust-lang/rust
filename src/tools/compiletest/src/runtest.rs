// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use common::Config;
use common::{CompileFail, ParseFail, Pretty, RunFail, RunPass, RunPassValgrind};
use common::{Codegen, DebugInfoLldb, DebugInfoGdb, Rustdoc, CodegenUnits};
use common::{Incremental};
use errors::{self, ErrorKind, Error};
use json;
use header::TestProps;
use header;
use procsrv;
use test::TestPaths;
use util::logv;

use std::env;
use std::collections::HashSet;
use std::fmt;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::prelude::*;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, ExitStatus};

pub fn run(config: Config, testpaths: &TestPaths) {
    match &*config.target {

        "arm-linux-androideabi" | "aarch64-linux-android" => {
            if !config.adb_device_status {
                panic!("android device not available");
            }
        }

        _=> { }
    }

    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        print!("\n\n");
    }
    debug!("running {:?}", testpaths.file.display());
    let props = header::load_props(&testpaths.file);
    debug!("loaded props");
    match config.mode {
        CompileFail => run_cfail_test(&config, &props, &testpaths),
        ParseFail => run_cfail_test(&config, &props, &testpaths),
        RunFail => run_rfail_test(&config, &props, &testpaths),
        RunPass => run_rpass_test(&config, &props, &testpaths),
        RunPassValgrind => run_valgrind_test(&config, &props, &testpaths),
        Pretty => run_pretty_test(&config, &props, &testpaths),
        DebugInfoGdb => run_debuginfo_gdb_test(&config, &props, &testpaths),
        DebugInfoLldb => run_debuginfo_lldb_test(&config, &props, &testpaths),
        Codegen => run_codegen_test(&config, &props, &testpaths),
        Rustdoc => run_rustdoc_test(&config, &props, &testpaths),
        CodegenUnits => run_codegen_units_test(&config, &props, &testpaths),
        Incremental => run_incremental_test(&config, &props, &testpaths),
    }
}

fn get_output(props: &TestProps, proc_res: &ProcRes) -> String {
    if props.check_stdout {
        format!("{}{}", proc_res.stdout, proc_res.stderr)
    } else {
        proc_res.stderr.clone()
    }
}


fn for_each_revision<OP>(config: &Config, props: &TestProps, testpaths: &TestPaths,
                         mut op: OP)
    where OP: FnMut(&Config, &TestProps, &TestPaths, Option<&str>)
{
    if props.revisions.is_empty() {
        op(config, props, testpaths, None)
    } else {
        for revision in &props.revisions {
            let mut revision_props = props.clone();
            header::load_props_into(&mut revision_props,
                                    &testpaths.file,
                                    Some(&revision));
            revision_props.compile_flags.extend(vec![
                format!("--cfg"),
                format!("{}", revision),
            ]);
            op(config, &revision_props, testpaths, Some(revision));
        }
    }
}

fn run_cfail_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    for_each_revision(config, props, testpaths, run_cfail_test_revision);
}

fn run_cfail_test_revision(config: &Config,
                           props: &TestProps,
                           testpaths: &TestPaths,
                           revision: Option<&str>) {
    let proc_res = compile_test(config, props, testpaths);

    if proc_res.status.success() {
        fatal_proc_rec(
            revision,
            &format!("{} test compiled successfully!", config.mode)[..],
            &proc_res);
    }

    check_correct_failure_status(revision, &proc_res);

    if proc_res.status.success() {
        fatal(revision, "process did not return an error status");
    }

    let output_to_check = get_output(props, &proc_res);
    let expected_errors = errors::load_errors(&testpaths.file, revision);
    if !expected_errors.is_empty() {
        if !props.error_patterns.is_empty() {
            fatal(revision, "both error pattern and expected errors specified");
        }
        check_expected_errors(revision, expected_errors, testpaths, &proc_res);
    } else {
        check_error_patterns(revision, props, testpaths, &output_to_check, &proc_res);
    }
    check_no_compiler_crash(revision, &proc_res);
    check_forbid_output(revision, props, &output_to_check, &proc_res);
}

fn run_rfail_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    for_each_revision(config, props, testpaths, run_rfail_test_revision);
}

fn run_rfail_test_revision(config: &Config,
                           props: &TestProps,
                           testpaths: &TestPaths,
                           revision: Option<&str>) {
    let proc_res = compile_test(config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(revision, "compilation failed!", &proc_res);
    }

    let proc_res = exec_compiled_test(config, props, testpaths);

    // The value our Makefile configures valgrind to return on failure
    const VALGRIND_ERR: i32 = 100;
    if proc_res.status.code() == Some(VALGRIND_ERR) {
        fatal_proc_rec(revision, "run-fail test isn't valgrind-clean!", &proc_res);
    }

    let output_to_check = get_output(props, &proc_res);
    check_correct_failure_status(revision, &proc_res);
    check_error_patterns(revision, props, testpaths, &output_to_check, &proc_res);
}

fn check_correct_failure_status(revision: Option<&str>, proc_res: &ProcRes) {
    // The value the rust runtime returns on failure
    const RUST_ERR: i32 = 101;
    if proc_res.status.code() != Some(RUST_ERR) {
        fatal_proc_rec(
            revision,
            &format!("failure produced the wrong error: {}",
                     proc_res.status),
            proc_res);
    }
}

fn run_rpass_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    for_each_revision(config, props, testpaths, run_rpass_test_revision);
}

fn run_rpass_test_revision(config: &Config,
                           props: &TestProps,
                           testpaths: &TestPaths,
                           revision: Option<&str>) {
    let proc_res = compile_test(config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(revision, "compilation failed!", &proc_res);
    }

    let proc_res = exec_compiled_test(config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(revision, "test run failed!", &proc_res);
    }
}

fn run_valgrind_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    assert!(props.revisions.is_empty(), "revisions not relevant here");

    if config.valgrind_path.is_none() {
        assert!(!config.force_valgrind);
        return run_rpass_test(config, props, testpaths);
    }

    let mut proc_res = compile_test(config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(None, "compilation failed!", &proc_res);
    }

    let mut new_config = config.clone();
    new_config.runtool = new_config.valgrind_path.clone();
    proc_res = exec_compiled_test(&new_config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(None, "test run failed!", &proc_res);
    }
}

fn run_pretty_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    for_each_revision(config, props, testpaths, run_pretty_test_revision);
}

fn run_pretty_test_revision(config: &Config,
                            props: &TestProps,
                            testpaths: &TestPaths,
                            revision: Option<&str>) {
    if props.pp_exact.is_some() {
        logv(config, "testing for exact pretty-printing".to_owned());
    } else {
        logv(config, "testing for converging pretty-printing".to_owned());
    }

    let rounds =
        match props.pp_exact { Some(_) => 1, None => 2 };

    let mut src = String::new();
    File::open(&testpaths.file).unwrap().read_to_string(&mut src).unwrap();
    let mut srcs = vec!(src);

    let mut round = 0;
    while round < rounds {
        logv(config, format!("pretty-printing round {} revision {:?}",
                             round, revision));
        let proc_res = print_source(config,
                                    props,
                                    testpaths,
                                    srcs[round].to_owned(),
                                    &props.pretty_mode);

        if !proc_res.status.success() {
            fatal_proc_rec(revision,
                           &format!("pretty-printing failed in round {} revision {:?}",
                                    round, revision),
                           &proc_res);
        }

        let ProcRes{ stdout, .. } = proc_res;
        srcs.push(stdout);
        round += 1;
    }

    let mut expected = match props.pp_exact {
        Some(ref file) => {
            let filepath = testpaths.file.parent().unwrap().join(file);
            let mut s = String::new();
            File::open(&filepath).unwrap().read_to_string(&mut s).unwrap();
            s
        }
        None => { srcs[srcs.len() - 2].clone() }
    };
    let mut actual = srcs[srcs.len() - 1].clone();

    if props.pp_exact.is_some() {
        // Now we have to care about line endings
        let cr = "\r".to_owned();
        actual = actual.replace(&cr, "").to_owned();
        expected = expected.replace(&cr, "").to_owned();
    }

    compare_source(revision, &expected, &actual);

    // If we're only making sure that the output matches then just stop here
    if props.pretty_compare_only { return; }

    // Finally, let's make sure it actually appears to remain valid code
    let proc_res = typecheck_source(config, props, testpaths, actual);
    if !proc_res.status.success() {
        fatal_proc_rec(revision, "pretty-printed source does not typecheck", &proc_res);
    }

    if !props.pretty_expanded { return }

    // additionally, run `--pretty expanded` and try to build it.
    let proc_res = print_source(config, props, testpaths, srcs[round].clone(), "expanded");
    if !proc_res.status.success() {
        fatal_proc_rec(revision, "pretty-printing (expanded) failed", &proc_res);
    }

    let ProcRes{ stdout: expanded_src, .. } = proc_res;
    let proc_res = typecheck_source(config, props, testpaths, expanded_src);
    if !proc_res.status.success() {
        fatal_proc_rec(
            revision,
            "pretty-printed source (expanded) does not typecheck",
            &proc_res);
    }

    return;

    fn print_source(config: &Config,
                    props: &TestProps,
                    testpaths: &TestPaths,
                    src: String,
                    pretty_type: &str) -> ProcRes {
        let aux_dir = aux_output_dir_name(config, testpaths);
        compose_and_run(config,
                        testpaths,
                        make_pp_args(config,
                                     props,
                                     testpaths,
                                     pretty_type.to_owned()),
                        props.exec_env.clone(),
                        config.compile_lib_path.to_str().unwrap(),
                        Some(aux_dir.to_str().unwrap()),
                        Some(src))
    }

    fn make_pp_args(config: &Config,
                    props: &TestProps,
                    testpaths: &TestPaths,
                    pretty_type: String) -> ProcArgs {
        let aux_dir = aux_output_dir_name(config, testpaths);
        // FIXME (#9639): This needs to handle non-utf8 paths
        let mut args = vec!("-".to_owned(),
                            "-Zunstable-options".to_owned(),
                            "--unpretty".to_owned(),
                            pretty_type,
                            format!("--target={}", config.target),
                            "-L".to_owned(),
                            aux_dir.to_str().unwrap().to_owned());
        args.extend(split_maybe_args(&config.target_rustcflags));
        args.extend(props.compile_flags.iter().cloned());
        return ProcArgs {
            prog: config.rustc_path.to_str().unwrap().to_owned(),
            args: args,
        };
    }

    fn compare_source(revision: Option<&str>, expected: &str, actual: &str) {
        if expected != actual {
            error(revision, "pretty-printed source does not match expected source");
            println!("\n\
expected:\n\
------------------------------------------\n\
{}\n\
------------------------------------------\n\
actual:\n\
------------------------------------------\n\
{}\n\
------------------------------------------\n\
\n",
                     expected, actual);
            panic!();
        }
    }

    fn typecheck_source(config: &Config, props: &TestProps,
                        testpaths: &TestPaths, src: String) -> ProcRes {
        let args = make_typecheck_args(config, props, testpaths);
        compose_and_run_compiler(config, props, testpaths, args, Some(src))
    }

    fn make_typecheck_args(config: &Config, props: &TestProps, testpaths: &TestPaths) -> ProcArgs {
        let aux_dir = aux_output_dir_name(config, testpaths);
        let target = if props.force_host {
            &*config.host
        } else {
            &*config.target
        };
        // FIXME (#9639): This needs to handle non-utf8 paths
        let mut args = vec!("-".to_owned(),
                            "-Zno-trans".to_owned(),
                            format!("--target={}", target),
                            "-L".to_owned(),
                            config.build_base.to_str().unwrap().to_owned(),
                            "-L".to_owned(),
                            aux_dir.to_str().unwrap().to_owned());
        args.extend(split_maybe_args(&config.target_rustcflags));
        args.extend(props.compile_flags.iter().cloned());
        // FIXME (#9639): This needs to handle non-utf8 paths
        return ProcArgs {
            prog: config.rustc_path.to_str().unwrap().to_owned(),
            args: args,
        };
    }
}

fn run_debuginfo_gdb_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    assert!(props.revisions.is_empty(), "revisions not relevant here");

    let mut config = Config {
        target_rustcflags: cleanup_debug_info_options(&config.target_rustcflags),
        host_rustcflags: cleanup_debug_info_options(&config.host_rustcflags),
        .. config.clone()
    };

    let config = &mut config;
    let DebuggerCommands {
        commands,
        check_lines,
        breakpoint_lines
    } = parse_debugger_commands(testpaths, "gdb");
    let mut cmds = commands.join("\n");

    // compile test file (it should have 'compile-flags:-g' in the header)
    let compiler_run_result = compile_test(config, props, testpaths);
    if !compiler_run_result.status.success() {
        fatal_proc_rec(None, "compilation failed!", &compiler_run_result);
    }

    let exe_file = make_exe_name(config, testpaths);

    let debugger_run_result;
    match &*config.target {
        "arm-linux-androideabi" | "aarch64-linux-android" => {

            cmds = cmds.replace("run", "continue");

            // write debugger script
            let mut script_str = String::with_capacity(2048);
            script_str.push_str(&format!("set charset {}\n", charset()));
            script_str.push_str(&format!("file {}\n", exe_file.to_str().unwrap()));
            script_str.push_str("target remote :5039\n");
            script_str.push_str(&format!("set solib-search-path \
                                         ./{}/stage2/lib/rustlib/{}/lib/\n",
                                         config.host, config.target));
            for line in &breakpoint_lines {
                script_str.push_str(&format!("break {:?}:{}\n",
                                             testpaths.file
                                                      .file_name()
                                                      .unwrap()
                                                      .to_string_lossy(),
                                             *line)[..]);
            }
            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testpaths,
                             &script_str,
                             "debugger.script");


            procsrv::run("",
                         &config.adb_path,
                         None,
                         &[
                            "push".to_owned(),
                            exe_file.to_str().unwrap().to_owned(),
                            config.adb_test_dir.clone()
                         ],
                         vec!(("".to_owned(), "".to_owned())),
                         Some("".to_owned()))
                .expect(&format!("failed to exec `{:?}`", config.adb_path));

            procsrv::run("",
                         &config.adb_path,
                         None,
                         &[
                            "forward".to_owned(),
                            "tcp:5039".to_owned(),
                            "tcp:5039".to_owned()
                         ],
                         vec!(("".to_owned(), "".to_owned())),
                         Some("".to_owned()))
                .expect(&format!("failed to exec `{:?}`", config.adb_path));

            let adb_arg = format!("export LD_LIBRARY_PATH={}; \
                                   gdbserver{} :5039 {}/{}",
                                  config.adb_test_dir.clone(),
                                  if config.target.contains("aarch64")
                                  {"64"} else {""},
                                  config.adb_test_dir.clone(),
                                  exe_file.file_name().unwrap().to_str()
                                          .unwrap());

            let mut process = procsrv::run_background("",
                                                      &config.adb_path
                                                            ,
                                                      None,
                                                      &[
                                                        "shell".to_owned(),
                                                        adb_arg.clone()
                                                      ],
                                                      vec!(("".to_owned(),
                                                            "".to_owned())),
                                                      Some("".to_owned()))
                .expect(&format!("failed to exec `{:?}`", config.adb_path));
            loop {
                //waiting 1 second for gdbserver start
                ::std::thread::sleep(::std::time::Duration::new(1,0));
                if TcpStream::connect("127.0.0.1:5039").is_ok() {
                    break
                }
            }

            let tool_path = match config.android_cross_path.to_str() {
                Some(x) => x.to_owned(),
                None => fatal(None, "cannot find android cross path")
            };

            let debugger_script = make_out_name(config, testpaths, "debugger.script");
            // FIXME (#9639): This needs to handle non-utf8 paths
            let debugger_opts =
                vec!("-quiet".to_owned(),
                     "-batch".to_owned(),
                     "-nx".to_owned(),
                     format!("-command={}", debugger_script.to_str().unwrap()));

            let mut gdb_path = tool_path;
            gdb_path.push_str(&format!("/bin/{}-gdb", config.target));
            let procsrv::Result {
                out,
                err,
                status
            } = procsrv::run("",
                             &gdb_path,
                             None,
                             &debugger_opts,
                             vec!(("".to_owned(), "".to_owned())),
                             None)
                .expect(&format!("failed to exec `{:?}`", gdb_path));
            let cmdline = {
                let cmdline = make_cmdline("",
                                           &format!("{}-gdb", config.target),
                                           &debugger_opts);
                logv(config, format!("executing {}", cmdline));
                cmdline
            };

            debugger_run_result = ProcRes {
                status: Status::Normal(status),
                stdout: out,
                stderr: err,
                cmdline: cmdline
            };
            if process.kill().is_err() {
                println!("Adb process is already finished.");
            }
        }

        _=> {
            let rust_src_root = find_rust_src_root(config)
                .expect("Could not find Rust source root");
            let rust_pp_module_rel_path = Path::new("./src/etc");
            let rust_pp_module_abs_path = rust_src_root.join(rust_pp_module_rel_path)
                                                       .to_str()
                                                       .unwrap()
                                                       .to_owned();
            // write debugger script
            let mut script_str = String::with_capacity(2048);
            script_str.push_str(&format!("set charset {}\n", charset()));
            script_str.push_str("show version\n");

            match config.gdb_version {
                Some(ref version) => {
                    println!("NOTE: compiletest thinks it is using GDB version {}",
                             version);

                    if header::gdb_version_to_int(version) >
                        header::gdb_version_to_int("7.4") {
                        // Add the directory containing the pretty printers to
                        // GDB's script auto loading safe path
                        script_str.push_str(
                            &format!("add-auto-load-safe-path {}\n",
                                     rust_pp_module_abs_path.replace(r"\", r"\\"))
                                );
                    }
                }
                _ => {
                    println!("NOTE: compiletest does not know which version of \
                              GDB it is using");
                }
            }

            // The following line actually doesn't have to do anything with
            // pretty printing, it just tells GDB to print values on one line:
            script_str.push_str("set print pretty off\n");

            // Add the pretty printer directory to GDB's source-file search path
            script_str.push_str(&format!("directory {}\n",
                                         rust_pp_module_abs_path));

            // Load the target executable
            script_str.push_str(&format!("file {}\n",
                                         exe_file.to_str().unwrap()
                                                 .replace(r"\", r"\\")));

            // Add line breakpoints
            for line in &breakpoint_lines {
                script_str.push_str(&format!("break '{}':{}\n",
                                             testpaths.file.file_name().unwrap()
                                                     .to_string_lossy(),
                                             *line));
            }

            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testpaths,
                             &script_str,
                             "debugger.script");

            // run debugger script with gdb
            fn debugger() -> &'static str {
                if cfg!(windows) {"gdb.exe"} else {"gdb"}
            }

            let debugger_script = make_out_name(config, testpaths, "debugger.script");

            // FIXME (#9639): This needs to handle non-utf8 paths
            let debugger_opts =
                vec!("-quiet".to_owned(),
                     "-batch".to_owned(),
                     "-nx".to_owned(),
                     format!("-command={}", debugger_script.to_str().unwrap()));

            let proc_args = ProcArgs {
                prog: debugger().to_owned(),
                args: debugger_opts,
            };

            let environment = vec![("PYTHONPATH".to_owned(), rust_pp_module_abs_path)];

            debugger_run_result = compose_and_run(config,
                                                  testpaths,
                                                  proc_args,
                                                  environment,
                                                  config.run_lib_path.to_str().unwrap(),
                                                  None,
                                                  None);
        }
    }

    if !debugger_run_result.status.success() {
        fatal(None, "gdb failed to execute");
    }

    check_debugger_output(&debugger_run_result, &check_lines);
}

fn find_rust_src_root(config: &Config) -> Option<PathBuf> {
    let mut path = config.src_base.clone();
    let path_postfix = Path::new("src/etc/lldb_batchmode.py");

    while path.pop() {
        if path.join(&path_postfix).is_file() {
            return Some(path);
        }
    }

    return None;
}

fn run_debuginfo_lldb_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    assert!(props.revisions.is_empty(), "revisions not relevant here");

    if config.lldb_python_dir.is_none() {
        fatal(None, "Can't run LLDB test because LLDB's python path is not set.");
    }

    let mut config = Config {
        target_rustcflags: cleanup_debug_info_options(&config.target_rustcflags),
        host_rustcflags: cleanup_debug_info_options(&config.host_rustcflags),
        .. config.clone()
    };

    let config = &mut config;

    // compile test file (it should have 'compile-flags:-g' in the header)
    let compile_result = compile_test(config, props, testpaths);
    if !compile_result.status.success() {
        fatal_proc_rec(None, "compilation failed!", &compile_result);
    }

    let exe_file = make_exe_name(config, testpaths);

    match config.lldb_version {
        Some(ref version) => {
            println!("NOTE: compiletest thinks it is using LLDB version {}",
                     version);
        }
        _ => {
            println!("NOTE: compiletest does not know which version of \
                      LLDB it is using");
        }
    }

    // Parse debugger commands etc from test files
    let DebuggerCommands {
        commands,
        check_lines,
        breakpoint_lines,
        ..
    } = parse_debugger_commands(testpaths, "lldb");

    // Write debugger script:
    // We don't want to hang when calling `quit` while the process is still running
    let mut script_str = String::from("settings set auto-confirm true\n");

    // Make LLDB emit its version, so we have it documented in the test output
    script_str.push_str("version\n");

    // Switch LLDB into "Rust mode"
    let rust_src_root = find_rust_src_root(config)
        .expect("Could not find Rust source root");
    let rust_pp_module_rel_path = Path::new("./src/etc/lldb_rust_formatters.py");
    let rust_pp_module_abs_path = rust_src_root.join(rust_pp_module_rel_path)
                                               .to_str()
                                               .unwrap()
                                               .to_owned();

    script_str.push_str(&format!("command script import {}\n",
                                 &rust_pp_module_abs_path[..])[..]);
    script_str.push_str("type summary add --no-value ");
    script_str.push_str("--python-function lldb_rust_formatters.print_val ");
    script_str.push_str("-x \".*\" --category Rust\n");
    script_str.push_str("type category enable Rust\n");

    // Set breakpoints on every line that contains the string "#break"
    let source_file_name = testpaths.file.file_name().unwrap().to_string_lossy();
    for line in &breakpoint_lines {
        script_str.push_str(&format!("breakpoint set --file '{}' --line {}\n",
                                     source_file_name,
                                     line));
    }

    // Append the other commands
    for line in &commands {
        script_str.push_str(line);
        script_str.push_str("\n");
    }

    // Finally, quit the debugger
    script_str.push_str("\nquit\n");

    // Write the script into a file
    debug!("script_str = {}", script_str);
    dump_output_file(config,
                     testpaths,
                     &script_str,
                     "debugger.script");
    let debugger_script = make_out_name(config, testpaths, "debugger.script");

    // Let LLDB execute the script via lldb_batchmode.py
    let debugger_run_result = run_lldb(config,
                                       testpaths,
                                       &exe_file,
                                       &debugger_script,
                                       &rust_src_root);

    if !debugger_run_result.status.success() {
        fatal_proc_rec(None, "Error while running LLDB", &debugger_run_result);
    }

    check_debugger_output(&debugger_run_result, &check_lines);

    fn run_lldb(config: &Config,
                testpaths: &TestPaths,
                test_executable: &Path,
                debugger_script: &Path,
                rust_src_root: &Path)
                -> ProcRes {
        // Prepare the lldb_batchmode which executes the debugger script
        let lldb_script_path = rust_src_root.join("src/etc/lldb_batchmode.py");
        cmd2procres(config,
                    testpaths,
                    Command::new(&config.python)
                            .arg(&lldb_script_path)
                            .arg(test_executable)
                            .arg(debugger_script)
                            .env("PYTHONPATH",
                                 config.lldb_python_dir.as_ref().unwrap()))
    }
}

fn cmd2procres(config: &Config, testpaths: &TestPaths, cmd: &mut Command)
              -> ProcRes {
    let (status, out, err) = match cmd.output() {
        Ok(Output { status, stdout, stderr }) => {
            (status,
             String::from_utf8(stdout).unwrap(),
             String::from_utf8(stderr).unwrap())
        },
        Err(e) => {
            fatal(None, &format!("Failed to setup Python process for \
                            LLDB script: {}", e))
        }
    };

    dump_output(config, testpaths, &out, &err);
    ProcRes {
        status: Status::Normal(status),
        stdout: out,
        stderr: err,
        cmdline: format!("{:?}", cmd)
    }
}

struct DebuggerCommands {
    commands: Vec<String>,
    check_lines: Vec<String>,
    breakpoint_lines: Vec<usize>,
}

fn parse_debugger_commands(testpaths: &TestPaths, debugger_prefix: &str)
                           -> DebuggerCommands {
    let command_directive = format!("{}-command", debugger_prefix);
    let check_directive = format!("{}-check", debugger_prefix);

    let mut breakpoint_lines = vec!();
    let mut commands = vec!();
    let mut check_lines = vec!();
    let mut counter = 1;
    let reader = BufReader::new(File::open(&testpaths.file).unwrap());
    for line in reader.lines() {
        match line {
            Ok(line) => {
                if line.contains("#break") {
                    breakpoint_lines.push(counter);
                }

                header::parse_name_value_directive(
                        &line,
                        &command_directive).map(|cmd| {
                    commands.push(cmd)
                });

                header::parse_name_value_directive(
                        &line,
                        &check_directive).map(|cmd| {
                    check_lines.push(cmd)
                });
            }
            Err(e) => {
                fatal(None, &format!("Error while parsing debugger commands: {}", e))
            }
        }
        counter += 1;
    }

    DebuggerCommands {
        commands: commands,
        check_lines: check_lines,
        breakpoint_lines: breakpoint_lines,
    }
}

fn cleanup_debug_info_options(options: &Option<String>) -> Option<String> {
    if options.is_none() {
        return None;
    }

    // Remove options that are either unwanted (-O) or may lead to duplicates due to RUSTFLAGS.
    let options_to_remove = [
        "-O".to_owned(),
        "-g".to_owned(),
        "--debuginfo".to_owned()
    ];
    let new_options =
        split_maybe_args(options).into_iter()
                                 .filter(|x| !options_to_remove.contains(x))
                                 .collect::<Vec<String>>();

    Some(new_options.join(" "))
}

fn check_debugger_output(debugger_run_result: &ProcRes, check_lines: &[String]) {
    let num_check_lines = check_lines.len();
    if num_check_lines > 0 {
        // Allow check lines to leave parts unspecified (e.g., uninitialized
        // bits in the wrong case of an enum) with the notation "[...]".
        let check_fragments: Vec<Vec<String>> =
            check_lines.iter().map(|s| {
                s
                 .trim()
                 .split("[...]")
                 .map(str::to_owned)
                 .collect()
            }).collect();
        // check if each line in props.check_lines appears in the
        // output (in order)
        let mut i = 0;
        for line in debugger_run_result.stdout.lines() {
            let mut rest = line.trim();
            let mut first = true;
            let mut failed = false;
            for frag in &check_fragments[i] {
                let found = if first {
                    if rest.starts_with(frag) {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    rest.find(frag)
                };
                match found {
                    None => {
                        failed = true;
                        break;
                    }
                    Some(i) => {
                        rest = &rest[(i + frag.len())..];
                    }
                }
                first = false;
            }
            if !failed && rest.is_empty() {
                i += 1;
            }
            if i == num_check_lines {
                // all lines checked
                break;
            }
        }
        if i != num_check_lines {
            fatal_proc_rec(None, &format!("line not found in debugger output: {}",
                                    check_lines.get(i).unwrap()),
                          debugger_run_result);
        }
    }
}

fn check_error_patterns(revision: Option<&str>,
                        props: &TestProps,
                        testpaths: &TestPaths,
                        output_to_check: &str,
                        proc_res: &ProcRes) {
    if props.error_patterns.is_empty() {
        fatal(revision,
              &format!("no error pattern specified in {:?}",
                       testpaths.file.display()));
    }
    let mut next_err_idx = 0;
    let mut next_err_pat = props.error_patterns[next_err_idx].trim();
    let mut done = false;
    for line in output_to_check.lines() {
        if line.contains(next_err_pat) {
            debug!("found error pattern {}", next_err_pat);
            next_err_idx += 1;
            if next_err_idx == props.error_patterns.len() {
                debug!("found all error patterns");
                done = true;
                break;
            }
            next_err_pat = props.error_patterns[next_err_idx].trim();
        }
    }
    if done { return; }

    let missing_patterns = &props.error_patterns[next_err_idx..];
    if missing_patterns.len() == 1 {
        fatal_proc_rec(
            revision,
            &format!("error pattern '{}' not found!", missing_patterns[0]),
            proc_res);
    } else {
        for pattern in missing_patterns {
            error(revision, &format!("error pattern '{}' not found!", *pattern));
        }
        fatal_proc_rec(revision, "multiple error patterns not found", proc_res);
    }
}

fn check_no_compiler_crash(revision: Option<&str>, proc_res: &ProcRes) {
    for line in proc_res.stderr.lines() {
        if line.starts_with("error: internal compiler error:") {
            fatal_proc_rec(revision,
                           "compiler encountered internal error",
                           proc_res);
        }
    }
}

fn check_forbid_output(revision: Option<&str>,
                       props: &TestProps,
                       output_to_check: &str,
                       proc_res: &ProcRes) {
    for pat in &props.forbid_output {
        if output_to_check.contains(pat) {
            fatal_proc_rec(revision,
                           "forbidden pattern found in compiler output",
                           proc_res);
        }
    }
}

fn check_expected_errors(revision: Option<&str>,
                         expected_errors: Vec<errors::Error>,
                         testpaths: &TestPaths,
                         proc_res: &ProcRes) {
    if proc_res.status.success() {
        fatal_proc_rec(revision, "process did not return an error status", proc_res);
    }

    let file_name =
        format!("{}", testpaths.file.display())
        .replace(r"\", "/"); // on windows, translate all '\' path separators to '/'

    // If the testcase being checked contains at least one expected "help"
    // message, then we'll ensure that all "help" messages are expected.
    // Otherwise, all "help" messages reported by the compiler will be ignored.
    // This logic also applies to "note" messages.
    let expect_help = expected_errors.iter().any(|ee| ee.kind == Some(ErrorKind::Help));
    let expect_note = expected_errors.iter().any(|ee| ee.kind == Some(ErrorKind::Note));

    // Parse the JSON output from the compiler and extract out the messages.
    let actual_errors = json::parse_output(&file_name, &proc_res.stderr);
    let mut unexpected = 0;
    let mut not_found = 0;
    let mut found = vec![false; expected_errors.len()];
    for actual_error in &actual_errors {
        let opt_index =
            expected_errors
            .iter()
            .enumerate()
            .position(|(index, expected_error)| {
                !found[index] &&
                    actual_error.line_num == expected_error.line_num &&
                    (expected_error.kind.is_none() ||
                     actual_error.kind == expected_error.kind) &&
                    actual_error.msg.contains(&expected_error.msg)
            });

        match opt_index {
            Some(index) => {
                // found a match, everybody is happy
                assert!(!found[index]);
                found[index] = true;
            }

            None => {
                if is_unexpected_compiler_message(actual_error,
                                                  expect_help,
                                                  expect_note) {
                    error(revision,
                          &format!("{}:{}: unexpected {:?}: '{}'",
                                   file_name,
                                   actual_error.line_num,
                                   actual_error.kind.as_ref()
                                                    .map_or(String::from("message"),
                                                            |k| k.to_string()),
                                   actual_error.msg));
                    unexpected += 1;
                }
            }
        }
    }

    // anything not yet found is a problem
    for (index, expected_error) in expected_errors.iter().enumerate() {
        if !found[index] {
            error(revision,
                  &format!("{}:{}: expected {} not found: {}",
                           file_name,
                           expected_error.line_num,
                           expected_error.kind.as_ref()
                                              .map_or("message".into(),
                                                      |k| k.to_string()),
                           expected_error.msg));
            not_found += 1;
        }
    }

    if unexpected > 0 || not_found > 0 {
        error(revision,
              &format!("{} unexpected errors found, {} expected errors not found",
                       unexpected, not_found));
        print!("status: {}\ncommand: {}\n",
               proc_res.status, proc_res.cmdline);
        println!("actual errors (from JSON output): {:#?}\n", actual_errors);
        println!("expected errors (from test file): {:#?}\n", expected_errors);
        panic!();
    }
}

/// Returns true if we should report an error about `actual_error`,
/// which did not match any of the expected error. We always require
/// errors/warnings to be explicitly listed, but only require
/// helps/notes if there are explicit helps/notes given.
fn is_unexpected_compiler_message(actual_error: &Error,
                                  expect_help: bool,
                                  expect_note: bool)
                                  -> bool {
    match actual_error.kind {
        Some(ErrorKind::Help) => expect_help,
        Some(ErrorKind::Note) => expect_note,
        Some(ErrorKind::Error) => true,
        Some(ErrorKind::Warning) => true,
        Some(ErrorKind::Suggestion) => false,
        None => false
    }
}

struct ProcArgs {
    prog: String,
    args: Vec<String>,
}

struct ProcRes {
    status: Status,
    stdout: String,
    stderr: String,
    cmdline: String,
}

enum Status {
    Parsed(i32),
    Normal(ExitStatus),
}

impl Status {
    fn code(&self) -> Option<i32> {
        match *self {
            Status::Parsed(i) => Some(i),
            Status::Normal(ref e) => e.code(),
        }
    }

    fn success(&self) -> bool {
        match *self {
            Status::Parsed(i) => i == 0,
            Status::Normal(ref e) => e.success(),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Status::Parsed(i) => write!(f, "exit code: {}", i),
            Status::Normal(ref e) => e.fmt(f),
        }
    }
}

fn compile_test(config: &Config, props: &TestProps,
                testpaths: &TestPaths) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testpaths);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let link_args = vec!("-L".to_owned(),
                         aux_dir.to_str().unwrap().to_owned());
    let args = make_compile_args(config,
                                 props,
                                 link_args,
                                 |a, b| TargetLocation::ThisFile(make_exe_name(a, b)), testpaths);
    compose_and_run_compiler(config, props, testpaths, args, None)
}

fn document(config: &Config,
            props: &TestProps,
            testpaths: &TestPaths,
            out_dir: &Path)
            -> ProcRes {
    if props.build_aux_docs {
        for rel_ab in &props.aux_builds {
            let aux_testpaths = compute_aux_test_paths(config, testpaths, rel_ab);
            let aux_props = header::load_props(&aux_testpaths.file);
            let auxres = document(config, &aux_props, &aux_testpaths, out_dir);
            if !auxres.status.success() {
                return auxres;
            }
        }
    }

    let aux_dir = aux_output_dir_name(config, testpaths);
    let mut args = vec!["-L".to_owned(),
                        aux_dir.to_str().unwrap().to_owned(),
                        "-o".to_owned(),
                        out_dir.to_str().unwrap().to_owned(),
                        testpaths.file.to_str().unwrap().to_owned()];
    args.extend(props.compile_flags.iter().cloned());
    let args = ProcArgs {
        prog: config.rustdoc_path.to_str().unwrap().to_owned(),
        args: args,
    };
    compose_and_run_compiler(config, props, testpaths, args, None)
}

fn exec_compiled_test(config: &Config, props: &TestProps,
                      testpaths: &TestPaths) -> ProcRes {

    let env = props.exec_env.clone();

    match &*config.target {

        "arm-linux-androideabi" | "aarch64-linux-android" => {
            _arm_exec_compiled_test(config, props, testpaths, env)
        }

        _=> {
            let aux_dir = aux_output_dir_name(config, testpaths);
            compose_and_run(config,
                            testpaths,
                            make_run_args(config, props, testpaths),
                            env,
                            config.run_lib_path.to_str().unwrap(),
                            Some(aux_dir.to_str().unwrap()),
                            None)
        }
    }
}

fn compute_aux_test_paths(config: &Config,
                          testpaths: &TestPaths,
                          rel_ab: &str)
                          -> TestPaths
{
    let abs_ab = config.aux_base.join(rel_ab);
    TestPaths {
        file: abs_ab,
        base: testpaths.base.clone(),
        relative_dir: Path::new(rel_ab).parent()
                                       .map(|p| p.to_path_buf())
                                       .unwrap_or_else(|| PathBuf::new())
    }
}

fn compose_and_run_compiler(config: &Config, props: &TestProps,
                            testpaths: &TestPaths, args: ProcArgs,
                            input: Option<String>) -> ProcRes {
    if !props.aux_builds.is_empty() {
        ensure_dir(&aux_output_dir_name(config, testpaths));
    }

    let aux_dir = aux_output_dir_name(config, testpaths);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let extra_link_args = vec!["-L".to_owned(),
                               aux_dir.to_str().unwrap().to_owned()];

    for rel_ab in &props.aux_builds {
        let aux_testpaths = compute_aux_test_paths(config, testpaths, rel_ab);
        let aux_props = header::load_props(&aux_testpaths.file);
        let mut crate_type = if aux_props.no_prefer_dynamic {
            Vec::new()
        } else {
            // We primarily compile all auxiliary libraries as dynamic libraries
            // to avoid code size bloat and large binaries as much as possible
            // for the test suite (otherwise including libstd statically in all
            // executables takes up quite a bit of space).
            //
            // For targets like MUSL or Emscripten, however, there is no support for
            // dynamic libraries so we just go back to building a normal library. Note,
            // however, that for MUSL if the library is built with `force_host` then
            // it's ok to be a dylib as the host should always support dylibs.
            if (config.target.contains("musl") && !aux_props.force_host) ||
                config.target.contains("emscripten")
            {
                vec!("--crate-type=lib".to_owned())
            } else {
                vec!("--crate-type=dylib".to_owned())
            }
        };
        crate_type.extend(extra_link_args.clone());
        let aux_args =
            make_compile_args(config,
                              &aux_props,
                              crate_type,
                              |a,b| {
                                  let f = make_lib_name(a, &b.file, testpaths);
                                  let parent = f.parent().unwrap();
                                  TargetLocation::ThisDirectory(parent.to_path_buf())
                              },
                              &aux_testpaths);
        let auxres = compose_and_run(config,
                                     &aux_testpaths,
                                     aux_args,
                                     Vec::new(),
                                     config.compile_lib_path.to_str().unwrap(),
                                     Some(aux_dir.to_str().unwrap()),
                                     None);
        if !auxres.status.success() {
            fatal_proc_rec(
                None,
                &format!("auxiliary build of {:?} failed to compile: ",
                        aux_testpaths.file.display()),
                &auxres);
        }

        match &*config.target {
            "arm-linux-androideabi"  | "aarch64-linux-android" => {
                _arm_push_aux_shared_library(config, testpaths);
            }
            _ => {}
        }
    }

    compose_and_run(config,
                    testpaths,
                    args,
                    props.rustc_env.clone(),
                    config.compile_lib_path.to_str().unwrap(),
                    Some(aux_dir.to_str().unwrap()),
                    input)
}

fn ensure_dir(path: &Path) {
    if path.is_dir() { return; }
    fs::create_dir_all(path).unwrap();
}

fn compose_and_run(config: &Config,
                   testpaths: &TestPaths,
                   ProcArgs{ args, prog }: ProcArgs,
                   procenv: Vec<(String, String)> ,
                   lib_path: &str,
                   aux_path: Option<&str>,
                   input: Option<String>) -> ProcRes {
    return program_output(config, testpaths, lib_path,
                          prog, aux_path, args, procenv, input);
}

enum TargetLocation {
    ThisFile(PathBuf),
    ThisDirectory(PathBuf),
}

fn make_compile_args<F>(config: &Config,
                        props: &TestProps,
                        extras: Vec<String> ,
                        xform: F,
                        testpaths: &TestPaths)
                        -> ProcArgs where
    F: FnOnce(&Config, &TestPaths) -> TargetLocation,
{
    let xform_file = xform(config, testpaths);
    let target = if props.force_host {
        &*config.host
    } else {
        &*config.target
    };
    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut args = vec!(testpaths.file.to_str().unwrap().to_owned(),
                        "-L".to_owned(),
                        config.build_base.to_str().unwrap().to_owned(),
                        format!("--target={}", target));

    match config.mode {
        CompileFail |
        ParseFail |
        Incremental => {
            // If we are extracting and matching errors in the new
            // fashion, then you want JSON mode. Old-skool error
            // patterns still match the raw compiler output.
            if props.error_patterns.is_empty() {
                args.extend(["--error-format",
                             "json",
                             "-Z",
                             "unstable-options"]
                            .iter()
                            .map(|s| s.to_string()));
            }
        }

        RunFail |
        RunPass |
        RunPassValgrind |
        Pretty |
        DebugInfoGdb |
        DebugInfoLldb |
        Codegen |
        Rustdoc |
        CodegenUnits => {
            // do not use JSON output
        }
    }

    args.extend_from_slice(&extras);
    if !props.no_prefer_dynamic {
        args.push("-C".to_owned());
        args.push("prefer-dynamic".to_owned());
    }
    let path = match xform_file {
        TargetLocation::ThisFile(path) => {
            args.push("-o".to_owned());
            path
        }
        TargetLocation::ThisDirectory(path) => {
            args.push("--out-dir".to_owned());
            path
        }
    };
    args.push(path.to_str().unwrap().to_owned());
    if props.force_host {
        args.extend(split_maybe_args(&config.host_rustcflags));
    } else {
        args.extend(split_maybe_args(&config.target_rustcflags));
    }
    args.extend(props.compile_flags.iter().cloned());
    return ProcArgs {
        prog: config.rustc_path.to_str().unwrap().to_owned(),
        args: args,
    };
}

fn make_lib_name(config: &Config, auxfile: &Path, testpaths: &TestPaths) -> PathBuf {
    // what we return here is not particularly important, as it
    // happens; rustc ignores everything except for the directory.
    let auxname = output_testname(auxfile);
    aux_output_dir_name(config, testpaths).join(&auxname)
}

fn make_exe_name(config: &Config, testpaths: &TestPaths) -> PathBuf {
    let mut f = output_base_name(config, testpaths);
    // FIXME: This is using the host architecture exe suffix, not target!
    if config.target == "asmjs-unknown-emscripten" {
        let mut fname = f.file_name().unwrap().to_os_string();
        fname.push(".js");
        f.set_file_name(&fname);
    } else if !env::consts::EXE_SUFFIX.is_empty() {
        let mut fname = f.file_name().unwrap().to_os_string();
        fname.push(env::consts::EXE_SUFFIX);
        f.set_file_name(&fname);
    }
    f
}

fn make_run_args(config: &Config, props: &TestProps, testpaths: &TestPaths)
                 -> ProcArgs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    let mut args = split_maybe_args(&config.runtool);

    // If this is emscripten, then run tests under nodejs
    if config.target == "asmjs-unknown-emscripten" {
        args.push("nodejs".to_owned());
    }

    let exe_file = make_exe_name(config, testpaths);

    // FIXME (#9639): This needs to handle non-utf8 paths
    args.push(exe_file.to_str().unwrap().to_owned());

    // Add the arguments in the run_flags directive
    args.extend(split_maybe_args(&props.run_flags));

    let prog = args.remove(0);
    return ProcArgs {
        prog: prog,
        args: args,
    };
}

fn split_maybe_args(argstr: &Option<String>) -> Vec<String> {
    match *argstr {
        Some(ref s) => {
            s
             .split(' ')
             .filter_map(|s| {
                 if s.chars().all(|c| c.is_whitespace()) {
                     None
                 } else {
                     Some(s.to_owned())
                 }
             }).collect()
        }
        None => Vec::new()
    }
}

fn program_output(config: &Config, testpaths: &TestPaths, lib_path: &str, prog: String,
                  aux_path: Option<&str>, args: Vec<String>,
                  env: Vec<(String, String)>,
                  input: Option<String>) -> ProcRes {
    let cmdline =
        {
            let cmdline = make_cmdline(lib_path,
                                       &prog,
                                       &args);
            logv(config, format!("executing {}", cmdline));
            cmdline
        };
    let procsrv::Result {
        out,
        err,
        status
    } = procsrv::run(lib_path,
                     &prog,
                     aux_path,
                     &args,
                     env,
                     input).expect(&format!("failed to exec `{}`", prog));
    dump_output(config, testpaths, &out, &err);
    return ProcRes {
        status: Status::Normal(status),
        stdout: out,
        stderr: err,
        cmdline: cmdline,
    };
}

fn make_cmdline(libpath: &str, prog: &str, args: &[String]) -> String {
    use util;

    // Linux and mac don't require adjusting the library search path
    if cfg!(unix) {
        format!("{} {}", prog, args.join(" "))
    } else {
        // Build the LD_LIBRARY_PATH variable as it would be seen on the command line
        // for diagnostic purposes
        fn lib_path_cmd_prefix(path: &str) -> String {
            format!("{}=\"{}\"", util::lib_path_env_var(), util::make_new_path(path))
        }

        format!("{} {} {}", lib_path_cmd_prefix(libpath), prog, args.join(" "))
    }
}

fn dump_output(config: &Config, testpaths: &TestPaths, out: &str, err: &str) {
    dump_output_file(config, testpaths, out, "out");
    dump_output_file(config, testpaths, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

fn dump_output_file(config: &Config,
                    testpaths: &TestPaths,
                    out: &str,
                    extension: &str) {
    let outfile = make_out_name(config, testpaths, extension);
    File::create(&outfile).unwrap().write_all(out.as_bytes()).unwrap();
}

fn make_out_name(config: &Config, testpaths: &TestPaths, extension: &str) -> PathBuf {
    output_base_name(config, testpaths).with_extension(extension)
}

fn aux_output_dir_name(config: &Config, testpaths: &TestPaths) -> PathBuf {
    let f = output_base_name(config, testpaths);
    let mut fname = f.file_name().unwrap().to_os_string();
    fname.push(&format!(".{}.libaux", config.mode));
    f.with_file_name(&fname)
}

fn output_testname(filepath: &Path) -> PathBuf {
    PathBuf::from(filepath.file_stem().unwrap())
}

fn output_base_name(config: &Config, testpaths: &TestPaths) -> PathBuf {
    let dir = config.build_base.join(&testpaths.relative_dir);

    // Note: The directory `dir` is created during `collect_tests_from_dir`
    dir
        .join(&output_testname(&testpaths.file))
        .with_extension(&config.stage_id)
}

fn maybe_dump_to_stdout(config: &Config, out: &str, err: &str) {
    if config.verbose {
        println!("------{}------------------------------", "stdout");
        println!("{}", out);
        println!("------{}------------------------------", "stderr");
        println!("{}", err);
        println!("------------------------------------------");
    }
}

fn error(revision: Option<&str>, err: &str) {
    match revision {
        Some(rev) => println!("\nerror in revision `{}`: {}", rev, err),
        None => println!("\nerror: {}", err)
    }
}

fn fatal(revision: Option<&str>, err: &str) -> ! {
    error(revision, err); panic!();
}

fn fatal_proc_rec(revision: Option<&str>, err: &str, proc_res: &ProcRes) -> ! {
    error(revision, err);
    print!("\
status: {}\n\
command: {}\n\
stdout:\n\
------------------------------------------\n\
{}\n\
------------------------------------------\n\
stderr:\n\
------------------------------------------\n\
{}\n\
------------------------------------------\n\
\n",
             proc_res.status, proc_res.cmdline, proc_res.stdout,
             proc_res.stderr);
    panic!();
}

fn _arm_exec_compiled_test(config: &Config,
                           props: &TestProps,
                           testpaths: &TestPaths,
                           env: Vec<(String, String)>)
                           -> ProcRes {
    let args = make_run_args(config, props, testpaths);
    let cmdline = make_cmdline("",
                               &args.prog,
                               &args.args);

    // get bare program string
    let mut tvec: Vec<String> = args.prog
                                    .split('/')
                                    .map(str::to_owned)
                                    .collect();
    let prog_short = tvec.pop().unwrap();

    // copy to target
    let copy_result = procsrv::run("",
                                   &config.adb_path,
                                   None,
                                   &[
                                    "push".to_owned(),
                                    args.prog.clone(),
                                    config.adb_test_dir.clone()
                                   ],
                                   vec!(("".to_owned(), "".to_owned())),
                                   Some("".to_owned()))
        .expect(&format!("failed to exec `{}`", config.adb_path));

    if config.verbose {
        println!("push ({}) {} {} {}",
                 config.target,
                 args.prog,
                 copy_result.out,
                 copy_result.err);
    }

    logv(config, format!("executing ({}) {}", config.target, cmdline));

    let mut runargs = Vec::new();

    // run test via adb_run_wrapper
    runargs.push("shell".to_owned());
    for (key, val) in env {
        runargs.push(format!("{}={}", key, val));
    }
    runargs.push(format!("{}/../adb_run_wrapper.sh", config.adb_test_dir));
    runargs.push(format!("{}", config.adb_test_dir));
    runargs.push(format!("{}", prog_short));

    for tv in &args.args {
        runargs.push(tv.to_owned());
    }
    procsrv::run("",
                 &config.adb_path,
                 None,
                 &runargs,
                 vec!(("".to_owned(), "".to_owned())), Some("".to_owned()))
        .expect(&format!("failed to exec `{}`", config.adb_path));

    // get exitcode of result
    runargs = Vec::new();
    runargs.push("shell".to_owned());
    runargs.push("cat".to_owned());
    runargs.push(format!("{}/{}.exitcode", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: exitcode_out, err: _, status: _ } =
        procsrv::run("",
                     &config.adb_path,
                     None,
                     &runargs,
                     vec!(("".to_owned(), "".to_owned())),
                     Some("".to_owned()))
        .expect(&format!("failed to exec `{}`", config.adb_path));

    let mut exitcode: i32 = 0;
    for c in exitcode_out.chars() {
        if !c.is_numeric() { break; }
        exitcode = exitcode * 10 + match c {
            '0' ... '9' => c as i32 - ('0' as i32),
            _ => 101,
        }
    }

    // get stdout of result
    runargs = Vec::new();
    runargs.push("shell".to_owned());
    runargs.push("cat".to_owned());
    runargs.push(format!("{}/{}.stdout", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stdout_out, err: _, status: _ } =
        procsrv::run("",
                     &config.adb_path,
                     None,
                     &runargs,
                     vec!(("".to_owned(), "".to_owned())),
                     Some("".to_owned()))
        .expect(&format!("failed to exec `{}`", config.adb_path));

    // get stderr of result
    runargs = Vec::new();
    runargs.push("shell".to_owned());
    runargs.push("cat".to_owned());
    runargs.push(format!("{}/{}.stderr", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stderr_out, err: _, status: _ } =
        procsrv::run("",
                     &config.adb_path,
                     None,
                     &runargs,
                     vec!(("".to_owned(), "".to_owned())),
                     Some("".to_owned()))
        .expect(&format!("failed to exec `{}`", config.adb_path));

    dump_output(config,
                testpaths,
                &stdout_out,
                &stderr_out);

    ProcRes {
        status: Status::Parsed(exitcode),
        stdout: stdout_out,
        stderr: stderr_out,
        cmdline: cmdline
    }
}

fn _arm_push_aux_shared_library(config: &Config, testpaths: &TestPaths) {
    let tdir = aux_output_dir_name(config, testpaths);

    let dirs = fs::read_dir(&tdir).unwrap();
    for file in dirs {
        let file = file.unwrap().path();
        if file.extension().and_then(|s| s.to_str()) == Some("so") {
            // FIXME (#9639): This needs to handle non-utf8 paths
            let copy_result = procsrv::run("",
                                           &config.adb_path,
                                           None,
                                           &[
                                            "push".to_owned(),
                                            file.to_str()
                                                .unwrap()
                                                .to_owned(),
                                            config.adb_test_dir.to_owned(),
                                           ],
                                           vec!(("".to_owned(),
                                                 "".to_owned())),
                                           Some("".to_owned()))
                .expect(&format!("failed to exec `{}`", config.adb_path));

            if config.verbose {
                println!("push ({}) {:?} {} {}",
                    config.target, file.display(),
                    copy_result.out, copy_result.err);
            }
        }
    }
}

// codegen tests (using FileCheck)

fn compile_test_and_save_ir(config: &Config, props: &TestProps,
                                 testpaths: &TestPaths) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testpaths);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut link_args = vec!("-L".to_owned(),
                             aux_dir.to_str().unwrap().to_owned());
    let llvm_args = vec!("--emit=llvm-ir".to_owned(),);
    link_args.extend(llvm_args);
    let args = make_compile_args(config,
                                 props,
                                 link_args,
                                 |a, b| TargetLocation::ThisDirectory(
                                     output_base_name(a, b).parent()
                                        .unwrap().to_path_buf()),
                                 testpaths);
    compose_and_run_compiler(config, props, testpaths, args, None)
}

fn check_ir_with_filecheck(config: &Config, testpaths: &TestPaths) -> ProcRes {
    let irfile = output_base_name(config, testpaths).with_extension("ll");
    let prog = config.llvm_filecheck.as_ref().unwrap();
    let proc_args = ProcArgs {
        // FIXME (#9639): This needs to handle non-utf8 paths
        prog: prog.to_str().unwrap().to_owned(),
        args: vec!(format!("-input-file={}", irfile.to_str().unwrap()),
                   testpaths.file.to_str().unwrap().to_owned())
    };
    compose_and_run(config, testpaths, proc_args, Vec::new(), "", None, None)
}

fn run_codegen_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    assert!(props.revisions.is_empty(), "revisions not relevant here");

    if config.llvm_filecheck.is_none() {
        fatal(None, "missing --llvm-filecheck");
    }

    let mut proc_res = compile_test_and_save_ir(config, props, testpaths);
    if !proc_res.status.success() {
        fatal_proc_rec(None, "compilation failed!", &proc_res);
    }

    proc_res = check_ir_with_filecheck(config, testpaths);
    if !proc_res.status.success() {
        fatal_proc_rec(None,
                       "verification with 'FileCheck' failed",
                       &proc_res);
    }
}

fn charset() -> &'static str {
    // FreeBSD 10.1 defaults to GDB 6.1.1 which doesn't support "auto" charset
    if cfg!(target_os = "bitrig") {
        "auto"
    } else if cfg!(target_os = "freebsd") {
        "ISO-8859-1"
    } else {
        "UTF-8"
    }
}

fn run_rustdoc_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    assert!(props.revisions.is_empty(), "revisions not relevant here");

    let out_dir = output_base_name(config, testpaths);
    let _ = fs::remove_dir_all(&out_dir);
    ensure_dir(&out_dir);

    let proc_res = document(config, props, testpaths, &out_dir);
    if !proc_res.status.success() {
        fatal_proc_rec(None, "rustdoc failed!", &proc_res);
    }
    let root = find_rust_src_root(config).unwrap();

    let res = cmd2procres(config,
                          testpaths,
                          Command::new(&config.python)
                                  .arg(root.join("src/etc/htmldocck.py"))
                                  .arg(out_dir)
                                  .arg(&testpaths.file));
    if !res.status.success() {
        fatal_proc_rec(None, "htmldocck failed!", &res);
    }
}

fn run_codegen_units_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {

    assert!(props.revisions.is_empty(), "revisions not relevant here");

    let proc_res = compile_test(config, props, testpaths);

    if !proc_res.status.success() {
        fatal_proc_rec(None, "compilation failed!", &proc_res);
    }

    check_no_compiler_crash(None, &proc_res);

    const PREFIX: &'static str = "TRANS_ITEM ";
    const CGU_MARKER: &'static str = "@@";

    let actual: Vec<TransItem> = proc_res
        .stdout
        .lines()
        .filter(|line| line.starts_with(PREFIX))
        .map(str_to_trans_item)
        .collect();

    let expected: Vec<TransItem> = errors::load_errors(&testpaths.file, None)
        .iter()
        .map(|e| str_to_trans_item(&e.msg[..]))
        .collect();

    let mut missing = Vec::new();
    let mut wrong_cgus = Vec::new();

    for expected_item in &expected {
        let actual_item_with_same_name = actual.iter()
                                               .find(|ti| ti.name == expected_item.name);

        if let Some(actual_item) = actual_item_with_same_name {
            if !expected_item.codegen_units.is_empty() {
                // Also check for codegen units
                if expected_item.codegen_units != actual_item.codegen_units {
                    wrong_cgus.push((expected_item.clone(), actual_item.clone()));
                }
            }
        } else {
            missing.push(expected_item.string.clone());
        }
    }

    let unexpected: Vec<_> =
        actual.iter()
              .filter(|acgu| !expected.iter().any(|ecgu| acgu.name == ecgu.name))
              .map(|acgu| acgu.string.clone())
              .collect();

    if !missing.is_empty() {
        missing.sort();

        println!("\nThese items should have been contained but were not:\n");

        for item in &missing {
            println!("{}", item);
        }

        println!("\n");
    }

    if !unexpected.is_empty() {
        let sorted = {
            let mut sorted = unexpected.clone();
            sorted.sort();
            sorted
        };

        println!("\nThese items were contained but should not have been:\n");

        for item in sorted {
            println!("{}", item);
        }

        println!("\n");
    }

    if !wrong_cgus.is_empty() {
        wrong_cgus.sort_by_key(|pair| pair.0.name.clone());
        println!("\nThe following items were assigned to wrong codegen units:\n");

        for &(ref expected_item, ref actual_item) in &wrong_cgus {
            println!("{}", expected_item.name);
            println!("  expected: {}", codegen_units_to_str(&expected_item.codegen_units));
            println!("  actual:   {}", codegen_units_to_str(&actual_item.codegen_units));
            println!("");
        }
    }

    if !(missing.is_empty() && unexpected.is_empty() && wrong_cgus.is_empty())
    {
        panic!();
    }

    #[derive(Clone, Eq, PartialEq)]
    struct TransItem {
        name: String,
        codegen_units: HashSet<String>,
        string: String,
    }

    // [TRANS_ITEM] name [@@ (cgu)+]
    fn str_to_trans_item(s: &str) -> TransItem {
        let s = if s.starts_with(PREFIX) {
            (&s[PREFIX.len()..]).trim()
        } else {
            s.trim()
        };

        let full_string = format!("{}{}", PREFIX, s.trim().to_owned());

        let parts: Vec<&str> = s.split(CGU_MARKER)
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .collect();

        let name = parts[0].trim();

        let cgus = if parts.len() > 1 {
            let cgus_str = parts[1];

            cgus_str.split(" ")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect()
        }
        else {
            HashSet::new()
        };

        TransItem {
            name: name.to_owned(),
            codegen_units: cgus,
            string: full_string,
        }
    }

    fn codegen_units_to_str(cgus: &HashSet<String>) -> String
    {
        let mut cgus: Vec<_> = cgus.iter().collect();
        cgus.sort();

        let mut string = String::new();
        for cgu in cgus {
            string.push_str(&cgu[..]);
            string.push_str(" ");
        }

        string
    }
}

fn run_incremental_test(config: &Config, props: &TestProps, testpaths: &TestPaths) {
    // Basic plan for a test incremental/foo/bar.rs:
    // - load list of revisions pass1, fail2, pass3
    //   - each should begin with `rpass`, `rfail`, or `cfail`
    //   - if `rpass`, expect compile and execution to succeed
    //   - if `cfail`, expect compilation to fail
    //   - if `rfail`, expect execution to fail
    // - create a directory build/foo/bar.incremental
    // - compile foo/bar.rs with -Z incremental=.../foo/bar.incremental and -C pass1
    //   - because name of revision starts with "pass", expect success
    // - compile foo/bar.rs with -Z incremental=.../foo/bar.incremental and -C fail2
    //   - because name of revision starts with "fail", expect an error
    //   - load expected errors as usual, but filter for those that end in `[fail2]`
    // - compile foo/bar.rs with -Z incremental=.../foo/bar.incremental and -C pass3
    //   - because name of revision starts with "pass", expect success
    // - execute build/foo/bar.exe and save output
    //
    // FIXME -- use non-incremental mode as an oracle? That doesn't apply
    // to #[rustc_dirty] and clean tests I guess

    assert!(!props.revisions.is_empty(), "incremental tests require a list of revisions");

    let output_base_name = output_base_name(config, testpaths);

    // Create the incremental workproduct directory.
    let incremental_dir = output_base_name.with_extension("incremental");
    if incremental_dir.exists() {
        fs::remove_dir_all(&incremental_dir).unwrap();
    }
    fs::create_dir_all(&incremental_dir).unwrap();

    if config.verbose {
        print!("incremental_dir={}", incremental_dir.display());
    }

    for revision in &props.revisions {
        let mut revision_props = props.clone();
        header::load_props_into(&mut revision_props, &testpaths.file, Some(&revision));

        revision_props.compile_flags.extend(vec![
            format!("-Z"),
            format!("incremental={}", incremental_dir.display()),
            format!("--cfg"),
            format!("{}", revision),
        ]);

        if config.verbose {
            print!("revision={:?} revision_props={:#?}", revision, revision_props);
        }

        if revision.starts_with("rpass") {
            run_rpass_test_revision(config, &revision_props, testpaths, Some(&revision));
        } else if revision.starts_with("rfail") {
            run_rfail_test_revision(config, &revision_props, testpaths, Some(&revision));
        } else if revision.starts_with("cfail") {
            run_cfail_test_revision(config, &revision_props, testpaths, Some(&revision));
        } else {
            fatal(
                Some(revision),
                "revision name must begin with rpass, rfail, or cfail");
        }
    }
}
