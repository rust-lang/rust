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
use common::{Codegen, DebugInfoLldb, DebugInfoGdb, Rustdoc};
use errors;
use header::TestProps;
use header;
use procsrv;
use util::logv;

use std::env;
use std::fmt;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::prelude::*;
use std::net::TcpStream;
use std::path::{Path, PathBuf, Component};
use std::process::{Command, Output, ExitStatus};

pub fn run(config: Config, testfile: &Path) {
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
    debug!("running {:?}", testfile.display());
    let props = header::load_props(&testfile);
    debug!("loaded props");
    match config.mode {
        CompileFail => run_cfail_test(&config, &props, &testfile),
        ParseFail => run_cfail_test(&config, &props, &testfile),
        RunFail => run_rfail_test(&config, &props, &testfile),
        RunPass => run_rpass_test(&config, &props, &testfile),
        RunPassValgrind => run_valgrind_test(&config, &props, &testfile),
        Pretty => run_pretty_test(&config, &props, &testfile),
        DebugInfoGdb => run_debuginfo_gdb_test(&config, &props, &testfile),
        DebugInfoLldb => run_debuginfo_lldb_test(&config, &props, &testfile),
        Codegen => run_codegen_test(&config, &props, &testfile),
        Rustdoc => run_rustdoc_test(&config, &props, &testfile),
    }
}

fn get_output(props: &TestProps, proc_res: &ProcRes) -> String {
    if props.check_stdout {
        format!("{}{}", proc_res.stdout, proc_res.stderr)
    } else {
        proc_res.stderr.clone()
    }
}

fn run_cfail_test(config: &Config, props: &TestProps, testfile: &Path) {
    let proc_res = compile_test(config, props, testfile);

    if proc_res.status.success() {
        fatal_proc_rec(&format!("{} test compiled successfully!", config.mode)[..],
                      &proc_res);
    }

    check_correct_failure_status(&proc_res);

    if proc_res.status.success() {
        fatal("process did not return an error status");
    }

    let output_to_check = get_output(props, &proc_res);
    let expected_errors = errors::load_errors(testfile);
    if !expected_errors.is_empty() {
        if !props.error_patterns.is_empty() {
            fatal("both error pattern and expected errors specified");
        }
        check_expected_errors(expected_errors, testfile, &proc_res);
    } else {
        check_error_patterns(props, testfile, &output_to_check, &proc_res);
    }
    check_no_compiler_crash(&proc_res);
    check_forbid_output(props, &output_to_check, &proc_res);
}

fn run_rfail_test(config: &Config, props: &TestProps, testfile: &Path) {
    let proc_res = compile_test(config, props, testfile);

    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    let proc_res = exec_compiled_test(config, props, testfile);

    // The value our Makefile configures valgrind to return on failure
    const VALGRIND_ERR: i32 = 100;
    if proc_res.status.code() == Some(VALGRIND_ERR) {
        fatal_proc_rec("run-fail test isn't valgrind-clean!", &proc_res);
    }

    let output_to_check = get_output(props, &proc_res);
    check_correct_failure_status(&proc_res);
    check_error_patterns(props, testfile, &output_to_check, &proc_res);
}

fn check_correct_failure_status(proc_res: &ProcRes) {
    // The value the rust runtime returns on failure
    const RUST_ERR: i32 = 101;
    if proc_res.status.code() != Some(RUST_ERR) {
        fatal_proc_rec(
            &format!("failure produced the wrong error: {}",
                     proc_res.status),
            proc_res);
    }
}

fn run_rpass_test(config: &Config, props: &TestProps, testfile: &Path) {
    let proc_res = compile_test(config, props, testfile);

    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    let proc_res = exec_compiled_test(config, props, testfile);

    if !proc_res.status.success() {
        fatal_proc_rec("test run failed!", &proc_res);
    }
}

fn run_valgrind_test(config: &Config, props: &TestProps, testfile: &Path) {
    if config.valgrind_path.is_none() {
        assert!(!config.force_valgrind);
        return run_rpass_test(config, props, testfile);
    }

    let mut proc_res = compile_test(config, props, testfile);

    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    let mut new_config = config.clone();
    new_config.runtool = new_config.valgrind_path.clone();
    proc_res = exec_compiled_test(&new_config, props, testfile);

    if !proc_res.status.success() {
        fatal_proc_rec("test run failed!", &proc_res);
    }
}

fn run_pretty_test(config: &Config, props: &TestProps, testfile: &Path) {
    if props.pp_exact.is_some() {
        logv(config, "testing for exact pretty-printing".to_owned());
    } else {
        logv(config, "testing for converging pretty-printing".to_owned());
    }

    let rounds =
        match props.pp_exact { Some(_) => 1, None => 2 };

    let mut src = String::new();
    File::open(testfile).unwrap().read_to_string(&mut src).unwrap();
    let mut srcs = vec!(src);

    let mut round = 0;
    while round < rounds {
        logv(config, format!("pretty-printing round {}", round));
        let proc_res = print_source(config,
                                    props,
                                    testfile,
                                    srcs[round].to_owned(),
                                    &props.pretty_mode);

        if !proc_res.status.success() {
            fatal_proc_rec(&format!("pretty-printing failed in round {}", round),
                          &proc_res);
        }

        let ProcRes{ stdout, .. } = proc_res;
        srcs.push(stdout);
        round += 1;
    }

    let mut expected = match props.pp_exact {
        Some(ref file) => {
            let filepath = testfile.parent().unwrap().join(file);
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

    compare_source(&expected, &actual);

    // If we're only making sure that the output matches then just stop here
    if props.pretty_compare_only { return; }

    // Finally, let's make sure it actually appears to remain valid code
    let proc_res = typecheck_source(config, props, testfile, actual);

    if !proc_res.status.success() {
        fatal_proc_rec("pretty-printed source does not typecheck", &proc_res);
    }
    if !props.pretty_expanded { return }

    // additionally, run `--pretty expanded` and try to build it.
    let proc_res = print_source(config, props, testfile, srcs[round].clone(), "expanded");
    if !proc_res.status.success() {
        fatal_proc_rec("pretty-printing (expanded) failed", &proc_res);
    }

    let ProcRes{ stdout: expanded_src, .. } = proc_res;
    let proc_res = typecheck_source(config, props, testfile, expanded_src);
    if !proc_res.status.success() {
        fatal_proc_rec("pretty-printed source (expanded) does not typecheck",
                      &proc_res);
    }

    return;

    fn print_source(config: &Config,
                    props: &TestProps,
                    testfile: &Path,
                    src: String,
                    pretty_type: &str) -> ProcRes {
        let aux_dir = aux_output_dir_name(config, testfile);
        compose_and_run(config,
                        testfile,
                        make_pp_args(config,
                                     props,
                                     testfile,
                                     pretty_type.to_owned()),
                        props.exec_env.clone(),
                        &config.compile_lib_path,
                        Some(aux_dir.to_str().unwrap()),
                        Some(src))
    }

    fn make_pp_args(config: &Config,
                    props: &TestProps,
                    testfile: &Path,
                    pretty_type: String) -> ProcArgs {
        let aux_dir = aux_output_dir_name(config, testfile);
        // FIXME (#9639): This needs to handle non-utf8 paths
        let mut args = vec!("-".to_owned(),
                            "-Zunstable-options".to_owned(),
                            "--unpretty".to_owned(),
                            pretty_type,
                            format!("--target={}", config.target),
                            "-L".to_owned(),
                            aux_dir.to_str().unwrap().to_owned());
        args.extend(split_maybe_args(&config.target_rustcflags));
        args.extend(split_maybe_args(&props.compile_flags));
        return ProcArgs {
            prog: config.rustc_path.to_str().unwrap().to_owned(),
            args: args,
        };
    }

    fn compare_source(expected: &str, actual: &str) {
        if expected != actual {
            error("pretty-printed source does not match expected source");
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
                        testfile: &Path, src: String) -> ProcRes {
        let args = make_typecheck_args(config, props, testfile);
        compose_and_run_compiler(config, props, testfile, args, Some(src))
    }

    fn make_typecheck_args(config: &Config, props: &TestProps, testfile: &Path) -> ProcArgs {
        let aux_dir = aux_output_dir_name(config, testfile);
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
        args.extend(split_maybe_args(&props.compile_flags));
        // FIXME (#9639): This needs to handle non-utf8 paths
        return ProcArgs {
            prog: config.rustc_path.to_str().unwrap().to_owned(),
            args: args,
        };
    }
}

fn run_debuginfo_gdb_test(config: &Config, props: &TestProps, testfile: &Path) {
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
    } = parse_debugger_commands(testfile, "gdb");
    let mut cmds = commands.join("\n");

    // compile test file (it should have 'compile-flags:-g' in the header)
    let compiler_run_result = compile_test(config, props, testfile);
    if !compiler_run_result.status.success() {
        fatal_proc_rec("compilation failed!", &compiler_run_result);
    }

    let exe_file = make_exe_name(config, testfile);

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
                                             testfile.file_name().unwrap()
                                                     .to_string_lossy(),
                                             *line)[..]);
            }
            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testfile,
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
                None => fatal("cannot find android cross path")
            };

            let debugger_script = make_out_name(config, testfile, "debugger.script");
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
                                             testfile.file_name().unwrap()
                                                     .to_string_lossy(),
                                             *line));
            }

            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testfile,
                             &script_str,
                             "debugger.script");

            // run debugger script with gdb
            fn debugger() -> &'static str {
                if cfg!(windows) {"gdb.exe"} else {"gdb"}
            }

            let debugger_script = make_out_name(config, testfile, "debugger.script");

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
                                                  testfile,
                                                  proc_args,
                                                  environment,
                                                  &config.run_lib_path,
                                                  None,
                                                  None);
        }
    }

    if !debugger_run_result.status.success() {
        fatal("gdb failed to execute");
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

fn run_debuginfo_lldb_test(config: &Config, props: &TestProps, testfile: &Path) {
    if config.lldb_python_dir.is_none() {
        fatal("Can't run LLDB test because LLDB's python path is not set.");
    }

    let mut config = Config {
        target_rustcflags: cleanup_debug_info_options(&config.target_rustcflags),
        host_rustcflags: cleanup_debug_info_options(&config.host_rustcflags),
        .. config.clone()
    };

    let config = &mut config;

    // compile test file (it should have 'compile-flags:-g' in the header)
    let compile_result = compile_test(config, props, testfile);
    if !compile_result.status.success() {
        fatal_proc_rec("compilation failed!", &compile_result);
    }

    let exe_file = make_exe_name(config, testfile);

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
    } = parse_debugger_commands(testfile, "lldb");

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
    for line in &breakpoint_lines {
        script_str.push_str(&format!("breakpoint set --line {}\n", line));
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
                     testfile,
                     &script_str,
                     "debugger.script");
    let debugger_script = make_out_name(config, testfile, "debugger.script");

    // Let LLDB execute the script via lldb_batchmode.py
    let debugger_run_result = run_lldb(config,
                                       &exe_file,
                                       &debugger_script,
                                       &rust_src_root);

    if !debugger_run_result.status.success() {
        fatal_proc_rec("Error while running LLDB", &debugger_run_result);
    }

    check_debugger_output(&debugger_run_result, &check_lines);

    fn run_lldb(config: &Config,
                test_executable: &Path,
                debugger_script: &Path,
                rust_src_root: &Path)
                -> ProcRes {
        // Prepare the lldb_batchmode which executes the debugger script
        let lldb_script_path = rust_src_root.join("src/etc/lldb_batchmode.py");
        cmd2procres(config,
                    test_executable,
                    Command::new(&config.python)
                            .arg(&lldb_script_path)
                            .arg(test_executable)
                            .arg(debugger_script)
                            .env("PYTHONPATH",
                                 config.lldb_python_dir.as_ref().unwrap()))
    }
}

fn cmd2procres(config: &Config, test_executable: &Path, cmd: &mut Command)
              -> ProcRes {
    let (status, out, err) = match cmd.output() {
        Ok(Output { status, stdout, stderr }) => {
            (status,
             String::from_utf8(stdout).unwrap(),
             String::from_utf8(stderr).unwrap())
        },
        Err(e) => {
            fatal(&format!("Failed to setup Python process for \
                            LLDB script: {}", e))
        }
    };

    dump_output(config, test_executable, &out, &err);
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

fn parse_debugger_commands(file_path: &Path, debugger_prefix: &str)
                           -> DebuggerCommands {
    let command_directive = format!("{}-command", debugger_prefix);
    let check_directive = format!("{}-check", debugger_prefix);

    let mut breakpoint_lines = vec!();
    let mut commands = vec!();
    let mut check_lines = vec!();
    let mut counter = 1;
    let reader = BufReader::new(File::open(file_path).unwrap());
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
                fatal(&format!("Error while parsing debugger commands: {}", e))
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
                                 .collect::<Vec<String>>()
                                 .join(" ");
    Some(new_options)
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
            fatal_proc_rec(&format!("line not found in debugger output: {}",
                                    check_lines.get(i).unwrap()),
                          debugger_run_result);
        }
    }
}

fn check_error_patterns(props: &TestProps,
                        testfile: &Path,
                        output_to_check: &str,
                        proc_res: &ProcRes) {
    if props.error_patterns.is_empty() {
        fatal(&format!("no error pattern specified in {:?}", testfile.display()));
    }
    let mut next_err_idx = 0;
    let mut next_err_pat = &props.error_patterns[next_err_idx];
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
            next_err_pat = &props.error_patterns[next_err_idx];
        }
    }
    if done { return; }

    let missing_patterns = &props.error_patterns[next_err_idx..];
    if missing_patterns.len() == 1 {
        fatal_proc_rec(&format!("error pattern '{}' not found!", missing_patterns[0]),
                      proc_res);
    } else {
        for pattern in missing_patterns {
            error(&format!("error pattern '{}' not found!", *pattern));
        }
        fatal_proc_rec("multiple error patterns not found", proc_res);
    }
}

fn check_no_compiler_crash(proc_res: &ProcRes) {
    for line in proc_res.stderr.lines() {
        if line.starts_with("error: internal compiler error:") {
            fatal_proc_rec("compiler encountered internal error",
                          proc_res);
        }
    }
}

fn check_forbid_output(props: &TestProps,
                       output_to_check: &str,
                       proc_res: &ProcRes) {
    for pat in &props.forbid_output {
        if output_to_check.contains(pat) {
            fatal_proc_rec("forbidden pattern found in compiler output", proc_res);
        }
    }
}

fn check_expected_errors(expected_errors: Vec<errors::ExpectedError>,
                         testfile: &Path,
                         proc_res: &ProcRes) {

    // true if we found the error in question
    let mut found_flags = vec![false; expected_errors.len()];

    if proc_res.status.success() {
        fatal("process did not return an error status");
    }

    let prefixes = expected_errors.iter().map(|ee| {
        format!("{}:{}:", testfile.display(), ee.line)
    }).collect::<Vec<String>>();

    fn prefix_matches(line: &str, prefix: &str) -> bool {
        use std::ascii::AsciiExt;
        // On windows just translate all '\' path separators to '/'
        let line = line.replace(r"\", "/");
        if cfg!(windows) {
            line.to_ascii_lowercase().starts_with(&prefix.to_ascii_lowercase())
        } else {
            line.starts_with(prefix)
        }
    }

    // A multi-line error will have followup lines which start with a space
    // or open paren.
    fn continuation( line: &str) -> bool {
        line.starts_with(" ") || line.starts_with("(")
    }

    // Scan and extract our error/warning messages,
    // which look like:
    //    filename:line1:col1: line2:col2: *error:* msg
    //    filename:line1:col1: line2:col2: *warning:* msg
    // where line1:col1: is the starting point, line2:col2:
    // is the ending point, and * represents ANSI color codes.
    //
    // This pattern is ambiguous on windows, because filename may contain
    // a colon, so any path prefix must be detected and removed first.
    for line in proc_res.stderr.lines() {
        let mut was_expected = false;
        let mut prev = 0;
        for (i, ee) in expected_errors.iter().enumerate() {
            if !found_flags[i] {
                debug!("prefix={} ee.kind={} ee.msg={} line={}",
                       prefixes[i],
                       ee.kind,
                       ee.msg,
                       line);
                // Suggestions have no line number in their output, so take on the line number of
                // the previous expected error
                if ee.kind == "suggestion" {
                    assert!(expected_errors[prev].kind == "help",
                            "SUGGESTIONs must be preceded by a HELP");
                    if line.contains(&ee.msg) {
                        found_flags[i] = true;
                        was_expected = true;
                        break;
                    }
                }
                if (prefix_matches(line, &prefixes[i]) || continuation(line)) &&
                    line.contains(&ee.kind) &&
                    line.contains(&ee.msg) {
                    found_flags[i] = true;
                    was_expected = true;
                    break;
                }
            }
            prev = i;
        }

        // ignore this msg which gets printed at the end
        if line.contains("aborting due to") {
            was_expected = true;
        }

        if !was_expected && is_compiler_error_or_warning(line) {
            fatal_proc_rec(&format!("unexpected compiler error or warning: '{}'",
                                    line),
                          proc_res);
        }
    }

    for (i, &flag) in found_flags.iter().enumerate() {
        if !flag {
            let ee = &expected_errors[i];
            fatal_proc_rec(&format!("expected {} on line {} not found: {}",
                                    ee.kind, ee.line, ee.msg),
                          proc_res);
        }
    }
}

fn is_compiler_error_or_warning(line: &str) -> bool {
    let mut c = Path::new(line).components();
    let line = match c.next() {
        Some(Component::Prefix(_)) => c.as_path().to_str().unwrap(),
        _ => line,
    };

    let mut i = 0;
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

fn scan_until_char(haystack: &str, needle: char, idx: &mut usize) -> bool {
    if *idx >= haystack.len() {
        return false;
    }
    let opt = haystack[(*idx)..].find(needle);
    if opt.is_none() {
        return false;
    }
    *idx = opt.unwrap();
    return true;
}

fn scan_char(haystack: &str, needle: char, idx: &mut usize) -> bool {
    if *idx >= haystack.len() {
        return false;
    }
    let ch = haystack.char_at(*idx);
    if ch != needle {
        return false;
    }
    *idx += ch.len_utf8();
    return true;
}

fn scan_integer(haystack: &str, idx: &mut usize) -> bool {
    let mut i = *idx;
    while i < haystack.len() {
        let ch = haystack.char_at(i);
        if ch < '0' || '9' < ch {
            break;
        }
        i += ch.len_utf8();
    }
    if i == *idx {
        return false;
    }
    *idx = i;
    return true;
}

fn scan_string(haystack: &str, needle: &str, idx: &mut usize) -> bool {
    let mut haystack_i = *idx;
    let mut needle_i = 0;
    while needle_i < needle.len() {
        if haystack_i >= haystack.len() {
            return false;
        }
        let ch = haystack.char_at(haystack_i);
        haystack_i += ch.len_utf8();
        if !scan_char(needle, ch, &mut needle_i) {
            return false;
        }
    }
    *idx = haystack_i;
    return true;
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
                testfile: &Path) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testfile);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let link_args = vec!("-L".to_owned(),
                         aux_dir.to_str().unwrap().to_owned());
    let args = make_compile_args(config,
                                 props,
                                 link_args,
                                 |a, b| TargetLocation::ThisFile(make_exe_name(a, b)), testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn document(config: &Config, props: &TestProps,
            testfile: &Path, out_dir: &Path) -> ProcRes {
    if props.build_aux_docs {
        for rel_ab in &props.aux_builds {
            let abs_ab = config.aux_base.join(rel_ab);
            let aux_props = header::load_props(&abs_ab);

            let auxres = document(config, &aux_props, &abs_ab, out_dir);
            if !auxres.status.success() {
                return auxres;
            }
        }
    }

    let aux_dir = aux_output_dir_name(config, testfile);
    let mut args = vec!["-L".to_owned(),
                        aux_dir.to_str().unwrap().to_owned(),
                        "-o".to_owned(),
                        out_dir.to_str().unwrap().to_owned(),
                        testfile.to_str().unwrap().to_owned()];
    args.extend(split_maybe_args(&props.compile_flags));
    let args = ProcArgs {
        prog: config.rustdoc_path.to_str().unwrap().to_owned(),
        args: args,
    };
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn exec_compiled_test(config: &Config, props: &TestProps,
                      testfile: &Path) -> ProcRes {

    let env = props.exec_env.clone();

    match &*config.target {

        "arm-linux-androideabi" | "aarch64-linux-android" => {
            _arm_exec_compiled_test(config, props, testfile, env)
        }

        _=> {
            let aux_dir = aux_output_dir_name(config, testfile);
            compose_and_run(config,
                            testfile,
                            make_run_args(config, props, testfile),
                            env,
                            &config.run_lib_path,
                            Some(aux_dir.to_str().unwrap()),
                            None)
        }
    }
}

fn compose_and_run_compiler(config: &Config, props: &TestProps,
                            testfile: &Path, args: ProcArgs,
                            input: Option<String>) -> ProcRes {
    if !props.aux_builds.is_empty() {
        ensure_dir(&aux_output_dir_name(config, testfile));
    }

    let aux_dir = aux_output_dir_name(config, testfile);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let extra_link_args = vec!["-L".to_owned(),
                               aux_dir.to_str().unwrap().to_owned()];

    for rel_ab in &props.aux_builds {
        let abs_ab = config.aux_base.join(rel_ab);
        let aux_props = header::load_props(&abs_ab);
        let mut crate_type = if aux_props.no_prefer_dynamic {
            Vec::new()
        } else {
            // We primarily compile all auxiliary libraries as dynamic libraries
            // to avoid code size bloat and large binaries as much as possible
            // for the test suite (otherwise including libstd statically in all
            // executables takes up quite a bit of space).
            //
            // For targets like MUSL, however, there is no support for dynamic
            // libraries so we just go back to building a normal library. Note,
            // however, that if the library is built with `force_host` then it's
            // ok to be a dylib as the host should always support dylibs.
            if config.target.contains("musl") && !aux_props.force_host {
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
                                  let f = make_lib_name(a, b, testfile);
                                  let parent = f.parent().unwrap();
                                  TargetLocation::ThisDirectory(parent.to_path_buf())
                              },
                              &abs_ab);
        let auxres = compose_and_run(config,
                                     &abs_ab,
                                     aux_args,
                                     Vec::new(),
                                     &config.compile_lib_path,
                                     Some(aux_dir.to_str().unwrap()),
                                     None);
        if !auxres.status.success() {
            fatal_proc_rec(
                &format!("auxiliary build of {:?} failed to compile: ",
                        abs_ab.display()),
                &auxres);
        }

        match &*config.target {
            "arm-linux-androideabi"  | "aarch64-linux-android" => {
                _arm_push_aux_shared_library(config, testfile);
            }
            _ => {}
        }
    }

    compose_and_run(config,
                    testfile,
                    args,
                    Vec::new(),
                    &config.compile_lib_path,
                    Some(aux_dir.to_str().unwrap()),
                    input)
}

fn ensure_dir(path: &Path) {
    if path.is_dir() { return; }
    fs::create_dir(path).unwrap();
}

fn compose_and_run(config: &Config, testfile: &Path,
                   ProcArgs{ args, prog }: ProcArgs,
                   procenv: Vec<(String, String)> ,
                   lib_path: &str,
                   aux_path: Option<&str>,
                   input: Option<String>) -> ProcRes {
    return program_output(config, testfile, lib_path,
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
                        testfile: &Path)
                        -> ProcArgs where
    F: FnOnce(&Config, &Path) -> TargetLocation,
{
    let xform_file = xform(config, testfile);
    let target = if props.force_host {
        &*config.host
    } else {
        &*config.target
    };
    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut args = vec!(testfile.to_str().unwrap().to_owned(),
                        "-L".to_owned(),
                        config.build_base.to_str().unwrap().to_owned(),
                        format!("--target={}", target));
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
    args.extend(split_maybe_args(&props.compile_flags));
    return ProcArgs {
        prog: config.rustc_path.to_str().unwrap().to_owned(),
        args: args,
    };
}

fn make_lib_name(config: &Config, auxfile: &Path, testfile: &Path) -> PathBuf {
    // what we return here is not particularly important, as it
    // happens; rustc ignores everything except for the directory.
    let auxname = output_testname(auxfile);
    aux_output_dir_name(config, testfile).join(&auxname)
}

fn make_exe_name(config: &Config, testfile: &Path) -> PathBuf {
    let mut f = output_base_name(config, testfile);
    if !env::consts::EXE_SUFFIX.is_empty() {
        let mut fname = f.file_name().unwrap().to_os_string();
        fname.push(env::consts::EXE_SUFFIX);
        f.set_file_name(&fname);
    }
    f
}

fn make_run_args(config: &Config, props: &TestProps, testfile: &Path)
                 -> ProcArgs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    let mut args = split_maybe_args(&config.runtool);
    let exe_file = make_exe_name(config, testfile);

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

fn program_output(config: &Config, testfile: &Path, lib_path: &str, prog: String,
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
    dump_output(config, testfile, &out, &err);
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

fn dump_output(config: &Config, testfile: &Path, out: &str, err: &str) {
    dump_output_file(config, testfile, out, "out");
    dump_output_file(config, testfile, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

fn dump_output_file(config: &Config, testfile: &Path,
                    out: &str, extension: &str) {
    let outfile = make_out_name(config, testfile, extension);
    File::create(&outfile).unwrap().write_all(out.as_bytes()).unwrap();
}

fn make_out_name(config: &Config, testfile: &Path, extension: &str) -> PathBuf {
    output_base_name(config, testfile).with_extension(extension)
}

fn aux_output_dir_name(config: &Config, testfile: &Path) -> PathBuf {
    let f = output_base_name(config, testfile);
    let mut fname = f.file_name().unwrap().to_os_string();
    fname.push(&format!(".{}.libaux", config.mode));
    f.with_file_name(&fname)
}

fn output_testname(testfile: &Path) -> PathBuf {
    PathBuf::from(testfile.file_stem().unwrap())
}

fn output_base_name(config: &Config, testfile: &Path) -> PathBuf {
    config.build_base
        .join(&output_testname(testfile))
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

fn error(err: &str) { println!("\nerror: {}", err); }

fn fatal(err: &str) -> ! { error(err); panic!(); }

fn fatal_proc_rec(err: &str, proc_res: &ProcRes) -> ! {
    print!("\n\
error: {}\n\
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
             err, proc_res.status, proc_res.cmdline, proc_res.stdout,
             proc_res.stderr);
    panic!();
}

fn _arm_exec_compiled_test(config: &Config,
                           props: &TestProps,
                           testfile: &Path,
                           env: Vec<(String, String)>)
                           -> ProcRes {
    let args = make_run_args(config, props, testfile);
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
                testfile,
                &stdout_out,
                &stderr_out);

    ProcRes {
        status: Status::Parsed(exitcode),
        stdout: stdout_out,
        stderr: stderr_out,
        cmdline: cmdline
    }
}

fn _arm_push_aux_shared_library(config: &Config, testfile: &Path) {
    let tdir = aux_output_dir_name(config, testfile);

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
                                 testfile: &Path) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testfile);
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
                                 testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn check_ir_with_filecheck(config: &Config, testfile: &Path) -> ProcRes {
    let irfile = output_base_name(config, testfile).with_extension("ll");
    let prog = config.llvm_bin_path.as_ref().unwrap().join("FileCheck");
    let proc_args = ProcArgs {
        // FIXME (#9639): This needs to handle non-utf8 paths
        prog: prog.to_str().unwrap().to_owned(),
        args: vec!(format!("-input-file={}", irfile.to_str().unwrap()),
                   testfile.to_str().unwrap().to_owned())
    };
    compose_and_run(config, testfile, proc_args, Vec::new(), "", None, None)
}

fn run_codegen_test(config: &Config, props: &TestProps, testfile: &Path) {

    if config.llvm_bin_path.is_none() {
        fatal("missing --llvm-bin-path");
    }

    let mut proc_res = compile_test_and_save_ir(config, props, testfile);
    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    proc_res = check_ir_with_filecheck(config, testfile);
    if !proc_res.status.success() {
        fatal_proc_rec("verification with 'FileCheck' failed",
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

fn run_rustdoc_test(config: &Config, props: &TestProps, testfile: &Path) {
    let out_dir = output_base_name(config, testfile);
    let _ = fs::remove_dir_all(&out_dir);
    ensure_dir(&out_dir);

    let proc_res = document(config, props, testfile, &out_dir);
    if !proc_res.status.success() {
        fatal_proc_rec("rustdoc failed!", &proc_res);
    }
    let root = find_rust_src_root(config).unwrap();

    let res = cmd2procres(config,
                          testfile,
                          Command::new(&config.python)
                                  .arg(root.join("src/etc/htmldocck.py"))
                                  .arg(out_dir)
                                  .arg(testfile));
    if !res.status.success() {
        fatal_proc_rec("htmldocck failed!", &res);
    }
}
