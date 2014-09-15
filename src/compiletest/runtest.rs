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
use common::{CompileFail, Pretty, RunFail, RunPass, DebugInfoGdb};
use common::{Codegen, DebugInfoLldb};
use errors;
use header::TestProps;
use header;
use procsrv;
use util::logv;
#[cfg(target_os = "windows")]
use util;

use std::io::File;
use std::io::fs::PathExtensions;
use std::io::fs;
use std::io::net::tcp;
use std::io::process::ProcessExit;
use std::io::process;
use std::io::timer;
use std::io;
use std::os;
use std::str;
use std::string::String;
use std::task;
use std::time::Duration;
use test::MetricMap;

pub fn run(config: Config, testfile: String) {

    match config.target.as_slice() {

        "arm-linux-androideabi" => {
            if !config.adb_device_status {
                fail!("android device not available");
            }
        }

        _=> { }
    }

    let mut _mm = MetricMap::new();
    run_metrics(config, testfile, &mut _mm);
}

pub fn run_metrics(config: Config, testfile: String, mm: &mut MetricMap) {
    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        print!("\n\n");
    }
    let testfile = Path::new(testfile);
    debug!("running {}", testfile.display());
    let props = header::load_props(&testfile);
    debug!("loaded props");
    match config.mode {
      CompileFail => run_cfail_test(&config, &props, &testfile),
      RunFail => run_rfail_test(&config, &props, &testfile),
      RunPass => run_rpass_test(&config, &props, &testfile),
      Pretty => run_pretty_test(&config, &props, &testfile),
      DebugInfoGdb => run_debuginfo_gdb_test(&config, &props, &testfile),
      DebugInfoLldb => run_debuginfo_lldb_test(&config, &props, &testfile),
      Codegen => run_codegen_test(&config, &props, &testfile, mm),
    }
}

fn run_cfail_test(config: &Config, props: &TestProps, testfile: &Path) {
    let proc_res = compile_test(config, props, testfile);

    if proc_res.status.success() {
        fatal_proc_rec("compile-fail test compiled successfully!",
                      &proc_res);
    }

    check_correct_failure_status(&proc_res);

    let expected_errors = errors::load_errors(&config.cfail_regex, testfile);
    if !expected_errors.is_empty() {
        if !props.error_patterns.is_empty() {
            fatal("both error pattern and expected errors specified");
        }
        check_expected_errors(expected_errors, testfile, &proc_res);
    } else {
        check_error_patterns(props, testfile, &proc_res);
    }
    check_no_compiler_crash(&proc_res);
}

fn run_rfail_test(config: &Config, props: &TestProps, testfile: &Path) {
    let proc_res = if !config.jit {
        let proc_res = compile_test(config, props, testfile);

        if !proc_res.status.success() {
            fatal_proc_rec("compilation failed!", &proc_res);
        }

        exec_compiled_test(config, props, testfile)
    } else {
        jit_test(config, props, testfile)
    };

    // The value our Makefile configures valgrind to return on failure
    static VALGRIND_ERR: int = 100;
    if proc_res.status.matches_exit_status(VALGRIND_ERR) {
        fatal_proc_rec("run-fail test isn't valgrind-clean!", &proc_res);
    }

    check_correct_failure_status(&proc_res);
    check_error_patterns(props, testfile, &proc_res);
}

fn check_correct_failure_status(proc_res: &ProcRes) {
    // The value the rust runtime returns on failure
    static RUST_ERR: int = 101;
    if !proc_res.status.matches_exit_status(RUST_ERR) {
        fatal_proc_rec(
            format!("failure produced the wrong error: {}",
                    proc_res.status).as_slice(),
            proc_res);
    }
}

fn run_rpass_test(config: &Config, props: &TestProps, testfile: &Path) {
    if !config.jit {
        let mut proc_res = compile_test(config, props, testfile);

        if !proc_res.status.success() {
            fatal_proc_rec("compilation failed!", &proc_res);
        }

        proc_res = exec_compiled_test(config, props, testfile);

        if !proc_res.status.success() {
            fatal_proc_rec("test run failed!", &proc_res);
        }
    } else {
        let proc_res = jit_test(config, props, testfile);

        if !proc_res.status.success() {
            fatal_proc_rec("jit failed!", &proc_res);
        }
    }
}

fn run_pretty_test(config: &Config, props: &TestProps, testfile: &Path) {
    if props.pp_exact.is_some() {
        logv(config, "testing for exact pretty-printing".to_string());
    } else {
        logv(config, "testing for converging pretty-printing".to_string());
    }

    let rounds =
        match props.pp_exact { Some(_) => 1, None => 2 };

    let src = File::open(testfile).read_to_end().unwrap();
    let src = String::from_utf8(src.clone()).unwrap();
    let mut srcs = vec!(src);

    let mut round = 0;
    while round < rounds {
        logv(config, format!("pretty-printing round {}", round));
        let proc_res = print_source(config,
                                    props,
                                    testfile,
                                    srcs[round].to_string(),
                                    props.pretty_mode.as_slice());

        if !proc_res.status.success() {
            fatal_proc_rec(format!("pretty-printing failed in round {}",
                                   round).as_slice(),
                          &proc_res);
        }

        let ProcRes{ stdout, .. } = proc_res;
        srcs.push(stdout);
        round += 1;
    }

    let mut expected = match props.pp_exact {
        Some(ref file) => {
            let filepath = testfile.dir_path().join(file);
            let s = File::open(&filepath).read_to_end().unwrap();
            String::from_utf8(s).unwrap()
        }
        None => { srcs[srcs.len() - 2u].clone() }
    };
    let mut actual = srcs[srcs.len() - 1u].clone();

    if props.pp_exact.is_some() {
        // Now we have to care about line endings
        let cr = "\r".to_string();
        actual = actual.replace(cr.as_slice(), "").to_string();
        expected = expected.replace(cr.as_slice(), "").to_string();
    }

    compare_source(expected.as_slice(), actual.as_slice());

    // If we're only making sure that the output matches then just stop here
    if props.pretty_compare_only { return; }

    // Finally, let's make sure it actually appears to remain valid code
    let proc_res = typecheck_source(config, props, testfile, actual);

    if !proc_res.status.success() {
        fatal_proc_rec("pretty-printed source does not typecheck", &proc_res);
    }
    if props.no_pretty_expanded { return }

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
                                     pretty_type.to_string()),
                        props.exec_env.clone(),
                        config.compile_lib_path.as_slice(),
                        Some(aux_dir.as_str().unwrap()),
                        Some(src))
    }

    fn make_pp_args(config: &Config,
                    props: &TestProps,
                    testfile: &Path,
                    pretty_type: String) -> ProcArgs {
        let aux_dir = aux_output_dir_name(config, testfile);
        // FIXME (#9639): This needs to handle non-utf8 paths
        let mut args = vec!("-".to_string(),
                            "--pretty".to_string(),
                            pretty_type,
                            format!("--target={}", config.target),
                            "-L".to_string(),
                            aux_dir.as_str().unwrap().to_string());
        args.push_all_move(split_maybe_args(&config.target_rustcflags));
        args.push_all_move(split_maybe_args(&props.compile_flags));
        return ProcArgs {
            prog: config.rustc_path.as_str().unwrap().to_string(),
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
            fail!();
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
            config.host.as_slice()
        } else {
            config.target.as_slice()
        };
        // FIXME (#9639): This needs to handle non-utf8 paths
        let mut args = vec!("-".to_string(),
                            "--no-trans".to_string(),
                            "--crate-type=lib".to_string(),
                            format!("--target={}", target),
                            "-L".to_string(),
                            config.build_base.as_str().unwrap().to_string(),
                            "-L".to_string(),
                            aux_dir.as_str().unwrap().to_string());
        args.push_all_move(split_maybe_args(&config.target_rustcflags));
        args.push_all_move(split_maybe_args(&props.compile_flags));
        // FIXME (#9639): This needs to handle non-utf8 paths
        return ProcArgs {
            prog: config.rustc_path.as_str().unwrap().to_string(),
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
        use_gdb_pretty_printer,
        ..
    } = parse_debugger_commands(testfile, "gdb");
    let mut cmds = commands.connect("\n");

    // compile test file (it should have 'compile-flags:-g' in the header)
    let compiler_run_result = compile_test(config, props, testfile);
    if !compiler_run_result.status.success() {
        fatal_proc_rec("compilation failed!", &compiler_run_result);
    }

    let exe_file = make_exe_name(config, testfile);

    let debugger_run_result;
    match config.target.as_slice() {
        "arm-linux-androideabi" => {

            cmds = cmds.replace("run", "continue").to_string();

            // write debugger script
            let script_str = ["set charset UTF-8".to_string(),
                              format!("file {}", exe_file.as_str().unwrap()
                                                         .to_string()),
                              "target remote :5039".to_string(),
                              cmds,
                              "quit".to_string()].connect("\n");
            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testfile,
                             script_str.as_slice(),
                             "debugger.script");


            procsrv::run("",
                         config.adb_path.as_slice(),
                         None,
                         [
                            "push".to_string(),
                            exe_file.as_str().unwrap().to_string(),
                            config.adb_test_dir.clone()
                         ],
                         vec!(("".to_string(), "".to_string())),
                         Some("".to_string()))
                .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

            procsrv::run("",
                         config.adb_path.as_slice(),
                         None,
                         [
                            "forward".to_string(),
                            "tcp:5039".to_string(),
                            "tcp:5039".to_string()
                         ],
                         vec!(("".to_string(), "".to_string())),
                         Some("".to_string()))
                .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

            let adb_arg = format!("export LD_LIBRARY_PATH={}; \
                                   gdbserver :5039 {}/{}",
                                  config.adb_test_dir.clone(),
                                  config.adb_test_dir.clone(),
                                  str::from_utf8(
                                      exe_file.filename()
                                      .unwrap()).unwrap());

            let mut process = procsrv::run_background("",
                                                      config.adb_path
                                                            .as_slice(),
                                                      None,
                                                      [
                                                        "shell".to_string(),
                                                        adb_arg.clone()
                                                      ],
                                                      vec!(("".to_string(),
                                                            "".to_string())),
                                                      Some("".to_string()))
                .expect(format!("failed to exec `{}`", config.adb_path).as_slice());
            loop {
                //waiting 1 second for gdbserver start
                timer::sleep(Duration::milliseconds(1000));
                let result = task::try(proc() {
                    tcp::TcpStream::connect("127.0.0.1", 5039).unwrap();
                });
                if result.is_err() {
                    continue;
                }
                break;
            }

            let tool_path = match config.android_cross_path.as_str() {
                Some(x) => x.to_string(),
                None => fatal("cannot find android cross path")
            };

            let debugger_script = make_out_name(config, testfile, "debugger.script");
            // FIXME (#9639): This needs to handle non-utf8 paths
            let debugger_opts =
                vec!("-quiet".to_string(),
                     "-batch".to_string(),
                     "-nx".to_string(),
                     format!("-command={}", debugger_script.as_str().unwrap()));

            let gdb_path = tool_path.append("/bin/arm-linux-androideabi-gdb");
            let procsrv::Result {
                out,
                err,
                status
            } = procsrv::run("",
                             gdb_path.as_slice(),
                             None,
                             debugger_opts.as_slice(),
                             vec!(("".to_string(), "".to_string())),
                             None)
                .expect(format!("failed to exec `{}`", gdb_path).as_slice());
            let cmdline = {
                let cmdline = make_cmdline("",
                                           "arm-linux-androideabi-gdb",
                                           debugger_opts.as_slice());
                logv(config, format!("executing {}", cmdline));
                cmdline
            };

            debugger_run_result = ProcRes {
                status: status,
                stdout: out,
                stderr: err,
                cmdline: cmdline
            };
            process.signal_kill().unwrap();
        }

        _=> {
            let rust_src_root = find_rust_src_root(config)
                .expect("Could not find Rust source root");
            let rust_pp_module_rel_path = Path::new("./src/etc");
            let rust_pp_module_abs_path = rust_src_root.join(rust_pp_module_rel_path)
                                                       .as_str()
                                                       .unwrap()
                                                       .to_string();
            // write debugger script
            let mut script_str = String::with_capacity(2048);

            script_str.push_str("set charset UTF-8\n");
            script_str.push_str("show version\n");

            match config.gdb_version {
                Some(ref version) => {
                    println!("NOTE: compiletest thinks it is using GDB version {}",
                             version.as_slice());

                    if header::gdb_version_to_int(version.as_slice()) >
                        header::gdb_version_to_int("7.4") {
                        // Add the directory containing the pretty printers to
                        // GDB's script auto loading safe path ...
                        script_str.push_str(
                            format!("add-auto-load-safe-path {}\n",
                                    rust_pp_module_abs_path.replace("\\", "\\\\").as_slice())
                                .as_slice());
                        // ... and also the test directory
                        script_str.push_str(
                            format!("add-auto-load-safe-path {}\n",
                                    config.build_base.as_str().unwrap().replace("\\", "\\\\"))
                                .as_slice());
                    }
                }
                _ => {
                    println!("NOTE: compiletest does not know which version of \
                              GDB it is using");
                }
            }

            // Load the target executable
            script_str.push_str(format!("file {}\n",
                                        exe_file.as_str().unwrap().replace("\\", "\\\\"))
                                    .as_slice());

            script_str.push_str(cmds.as_slice());
            script_str.push_str("quit\n");

            debug!("script_str = {}", script_str);
            dump_output_file(config,
                             testfile,
                             script_str.as_slice(),
                             "debugger.script");

            if use_gdb_pretty_printer {
                // Only emit the gdb auto-loading script if pretty printers
                // should actually be loaded
                dump_gdb_autoload_script(config, testfile);
            }

            // run debugger script with gdb
            #[cfg(windows)]
            fn debugger() -> String {
                "gdb.exe".to_string()
            }
            #[cfg(unix)]
            fn debugger() -> String {
                "gdb".to_string()
            }

            let debugger_script = make_out_name(config, testfile, "debugger.script");

            // FIXME (#9639): This needs to handle non-utf8 paths
            let debugger_opts =
                vec!("-quiet".to_string(),
                     "-batch".to_string(),
                     "-nx".to_string(),
                     format!("-command={}", debugger_script.as_str().unwrap()));

            let proc_args = ProcArgs {
                prog: debugger(),
                args: debugger_opts,
            };

            let environment = vec![("PYTHONPATH".to_string(), rust_pp_module_abs_path)];

            debugger_run_result = compose_and_run(config,
                                                  testfile,
                                                  proc_args,
                                                  environment,
                                                  config.run_lib_path.as_slice(),
                                                  None,
                                                  None);
        }
    }

    if !debugger_run_result.status.success() {
        fatal("gdb failed to execute");
    }

    check_debugger_output(&debugger_run_result, check_lines.as_slice());

    fn dump_gdb_autoload_script(config: &Config, testfile: &Path) {
        let mut script_path = output_base_name(config, testfile);
        let mut script_file_name = script_path.filename().unwrap().to_vec();
        script_file_name.push_all("-gdb.py".as_bytes());
        script_path.set_filename(script_file_name.as_slice());

        let script_content = "import gdb_rust_pretty_printing\n\
                              gdb_rust_pretty_printing.register_printers(gdb.current_objfile())\n"
                             .as_bytes();

        File::create(&script_path).write(script_content).unwrap();
    }
}

fn find_rust_src_root(config: &Config) -> Option<Path> {
    let mut path = config.src_base.clone();
    let path_postfix = Path::new("src/etc/lldb_batchmode.py");

    while path.pop() {
        if path.join(path_postfix.clone()).is_file() {
            return Some(path);
        }
    }

    return None;
}

fn run_debuginfo_lldb_test(config: &Config, props: &TestProps, testfile: &Path) {
    use std::io::process::{Command, ProcessOutput};

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

    // Parse debugger commands etc from test files
    let DebuggerCommands {
        commands,
        check_lines,
        breakpoint_lines,
        ..
    } = parse_debugger_commands(testfile, "lldb");

    // Write debugger script:
    // We don't want to hang when calling `quit` while the process is still running
    let mut script_str = String::from_str("settings set auto-confirm true\n");

    // Make LLDB emit its version, so we have it documented in the test output
    script_str.push_str("version\n");

    // Switch LLDB into "Rust mode"
    script_str.push_str("command script import ./src/etc/lldb_rust_formatters.py\n");
    script_str.push_str("type summary add --no-value ");
    script_str.push_str("--python-function lldb_rust_formatters.print_val ");
    script_str.push_str("-x \".*\" --category Rust\n");
    script_str.push_str("type category enable Rust\n");

    // Set breakpoints on every line that contains the string "#break"
    for line in breakpoint_lines.iter() {
        script_str.push_str(format!("breakpoint set --line {}\n",
                                    line).as_slice());
    }

    // Append the other commands
    for line in commands.iter() {
        script_str.push_str(line.as_slice());
        script_str.push_str("\n");
    }

    // Finally, quit the debugger
    script_str.push_str("quit\n");

    // Write the script into a file
    debug!("script_str = {}", script_str);
    dump_output_file(config,
                     testfile,
                     script_str.as_slice(),
                     "debugger.script");
    let debugger_script = make_out_name(config, testfile, "debugger.script");

    // Let LLDB execute the script via lldb_batchmode.py
    let debugger_run_result = run_lldb(config, &exe_file, &debugger_script);

    if !debugger_run_result.status.success() {
        fatal_proc_rec("Error while running LLDB", &debugger_run_result);
    }

    check_debugger_output(&debugger_run_result, check_lines.as_slice());

    fn run_lldb(config: &Config, test_executable: &Path, debugger_script: &Path) -> ProcRes {
        // Prepare the lldb_batchmode which executes the debugger script
        let mut cmd = Command::new("python");
        cmd.arg("./src/etc/lldb_batchmode.py")
           .arg(test_executable)
           .arg(debugger_script)
           .env_set_all([("PYTHONPATH", config.lldb_python_dir.clone().unwrap().as_slice())]);

        let (status, out, err) = match cmd.spawn() {
            Ok(process) => {
                let ProcessOutput { status, output, error } =
                    process.wait_with_output().unwrap();

                (status,
                 String::from_utf8(output).unwrap(),
                 String::from_utf8(error).unwrap())
            },
            Err(e) => {
                fatal(format!("Failed to setup Python process for \
                               LLDB script: {}", e).as_slice())
            }
        };

        dump_output(config, test_executable, out.as_slice(), err.as_slice());
        return ProcRes {
            status: status,
            stdout: out,
            stderr: err,
            cmdline: format!("{}", cmd)
        };
    }
}

struct DebuggerCommands {
    commands: Vec<String>,
    check_lines: Vec<String>,
    breakpoint_lines: Vec<uint>,
    use_gdb_pretty_printer: bool
}

fn parse_debugger_commands(file_path: &Path, debugger_prefix: &str)
                           -> DebuggerCommands {
    use std::io::{BufferedReader, File};

    let command_directive = format!("{}-command", debugger_prefix);
    let check_directive = format!("{}-check", debugger_prefix);

    let mut breakpoint_lines = vec!();
    let mut commands = vec!();
    let mut check_lines = vec!();
    let mut use_gdb_pretty_printer = false;
    let mut counter = 1;
    let mut reader = BufferedReader::new(File::open(file_path).unwrap());
    for line in reader.lines() {
        match line {
            Ok(line) => {
                if line.as_slice().contains("#break") {
                    breakpoint_lines.push(counter);
                }

                if line.as_slice().contains("gdb-use-pretty-printer") {
                    use_gdb_pretty_printer = true;
                }

                header::parse_name_value_directive(
                        line.as_slice(),
                        command_directive.as_slice()).map(|cmd| {
                    commands.push(cmd)
                });

                header::parse_name_value_directive(
                        line.as_slice(),
                        check_directive.as_slice()).map(|cmd| {
                    check_lines.push(cmd)
                });
            }
            Err(e) => {
                fatal(format!("Error while parsing debugger commands: {}",
                              e).as_slice())
            }
        }
        counter += 1;
    }

    DebuggerCommands {
        commands: commands,
        check_lines: check_lines,
        breakpoint_lines: breakpoint_lines,
        use_gdb_pretty_printer: use_gdb_pretty_printer,
    }
}

fn cleanup_debug_info_options(options: &Option<String>) -> Option<String> {
    if options.is_none() {
        return None;
    }

    // Remove options that are either unwanted (-O) or may lead to duplicates due to RUSTFLAGS.
    let options_to_remove = [
        "-O".to_string(),
        "-g".to_string(),
        "--debuginfo".to_string()
    ];
    let new_options =
        split_maybe_args(options).into_iter()
                                 .filter(|x| !options_to_remove.contains(x))
                                 .collect::<Vec<String>>()
                                 .connect(" ");
    Some(new_options)
}

fn check_debugger_output(debugger_run_result: &ProcRes, check_lines: &[String]) {
    let num_check_lines = check_lines.len();
    if num_check_lines > 0 {
        // Allow check lines to leave parts unspecified (e.g., uninitialized
        // bits in the wrong case of an enum) with the notation "[...]".
        let check_fragments: Vec<Vec<String>> =
            check_lines.iter().map(|s| {
                s.as_slice()
                 .trim()
                 .split_str("[...]")
                 .map(|x| x.to_string())
                 .collect()
            }).collect();
        // check if each line in props.check_lines appears in the
        // output (in order)
        let mut i = 0u;
        for line in debugger_run_result.stdout.as_slice().lines() {
            let mut rest = line.trim();
            let mut first = true;
            let mut failed = false;
            for frag in check_fragments[i].iter() {
                let found = if first {
                    if rest.starts_with(frag.as_slice()) {
                        Some(0)
                    } else {
                        None
                    }
                } else {
                    rest.find_str(frag.as_slice())
                };
                match found {
                    None => {
                        failed = true;
                        break;
                    }
                    Some(i) => {
                        rest = rest.slice_from(i + frag.len());
                    }
                }
                first = false;
            }
            if !failed && rest.len() == 0 {
                i += 1u;
            }
            if i == num_check_lines {
                // all lines checked
                break;
            }
        }
        if i != num_check_lines {
            fatal_proc_rec(format!("line not found in debugger output: {}",
                                  check_lines.get(i).unwrap()).as_slice(),
                          debugger_run_result);
        }
    }
}

fn check_error_patterns(props: &TestProps,
                        testfile: &Path,
                        proc_res: &ProcRes) {
    if props.error_patterns.is_empty() {
        fatal(format!("no error pattern specified in {}",
                      testfile.display()).as_slice());
    }

    if proc_res.status.success() {
        fatal("process did not return an error status");
    }

    let mut next_err_idx = 0u;
    let mut next_err_pat = &props.error_patterns[next_err_idx];
    let mut done = false;
    let output_to_check = if props.check_stdout {
        format!("{}{}", proc_res.stdout, proc_res.stderr)
    } else {
        proc_res.stderr.clone()
    };
    for line in output_to_check.as_slice().lines() {
        if line.contains(next_err_pat.as_slice()) {
            debug!("found error pattern {}", next_err_pat);
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
        fatal_proc_rec(format!("error pattern '{}' not found!",
                              missing_patterns[0]).as_slice(),
                      proc_res);
    } else {
        for pattern in missing_patterns.iter() {
            error(format!("error pattern '{}' not found!",
                          *pattern).as_slice());
        }
        fatal_proc_rec("multiple error patterns not found", proc_res);
    }
}

fn check_no_compiler_crash(proc_res: &ProcRes) {
    for line in proc_res.stderr.as_slice().lines() {
        if line.starts_with("error: internal compiler error:") {
            fatal_proc_rec("compiler encountered internal error",
                          proc_res);
        }
    }
}

fn check_expected_errors(expected_errors: Vec<errors::ExpectedError> ,
                         testfile: &Path,
                         proc_res: &ProcRes) {

    // true if we found the error in question
    let mut found_flags = Vec::from_elem(
        expected_errors.len(), false);

    if proc_res.status.success() {
        fatal("process did not return an error status");
    }

    let prefixes = expected_errors.iter().map(|ee| {
        format!("{}:{}:", testfile.display(), ee.line)
    }).collect::<Vec<String> >();

    #[cfg(target_os = "windows")]
    fn to_lower( s : &str ) -> String {
        let i = s.chars();
        let c : Vec<char> = i.map( |c| {
            if c.is_ascii() {
                c.to_ascii().to_lowercase().to_char()
            } else {
                c
            }
        } ).collect();
        String::from_chars(c.as_slice())
    }

    #[cfg(target_os = "windows")]
    fn prefix_matches( line : &str, prefix : &str ) -> bool {
        to_lower(line).as_slice().starts_with(to_lower(prefix).as_slice())
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    #[cfg(target_os = "dragonfly")]
    fn prefix_matches( line : &str, prefix : &str ) -> bool {
        line.starts_with( prefix )
    }

    // Scan and extract our error/warning messages,
    // which look like:
    //    filename:line1:col1: line2:col2: *error:* msg
    //    filename:line1:col1: line2:col2: *warning:* msg
    // where line1:col1: is the starting point, line2:col2:
    // is the ending point, and * represents ANSI color codes.
    for line in proc_res.stderr.as_slice().lines() {
        let mut was_expected = false;
        for (i, ee) in expected_errors.iter().enumerate() {
            if !found_flags[i] {
                debug!("prefix={} ee.kind={} ee.msg={} line={}",
                       prefixes[i].as_slice(),
                       ee.kind,
                       ee.msg,
                       line);
                if prefix_matches(line, prefixes[i].as_slice()) &&
                    line.contains(ee.kind.as_slice()) &&
                    line.contains(ee.msg.as_slice()) {
                    *found_flags.get_mut(i) = true;
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
            fatal_proc_rec(format!("unexpected compiler error or warning: '{}'",
                                  line).as_slice(),
                          proc_res);
        }
    }

    for (i, &flag) in found_flags.iter().enumerate() {
        if !flag {
            let ee = &expected_errors[i];
            fatal_proc_rec(format!("expected {} on line {} not found: {}",
                                  ee.kind, ee.line, ee.msg).as_slice(),
                          proc_res);
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
    *idx = opt.unwrap();
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

struct ProcArgs {
    prog: String,
    args: Vec<String>,
}

struct ProcRes {
    status: ProcessExit,
    stdout: String,
    stderr: String,
    cmdline: String,
}

fn compile_test(config: &Config, props: &TestProps,
                testfile: &Path) -> ProcRes {
    compile_test_(config, props, testfile, [])
}

fn jit_test(config: &Config, props: &TestProps, testfile: &Path) -> ProcRes {
    compile_test_(config, props, testfile, ["--jit".to_string()])
}

fn compile_test_(config: &Config, props: &TestProps,
                 testfile: &Path, extra_args: &[String]) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testfile);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let link_args = vec!("-L".to_string(),
                         aux_dir.as_str().unwrap().to_string());
    let args = make_compile_args(config,
                                 props,
                                 link_args.append(extra_args),
                                 |a, b| ThisFile(make_exe_name(a, b)), testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn exec_compiled_test(config: &Config, props: &TestProps,
                      testfile: &Path) -> ProcRes {

    let env = props.exec_env.clone();

    match config.target.as_slice() {

        "arm-linux-androideabi" => {
            _arm_exec_compiled_test(config, props, testfile, env)
        }

        _=> {
            let aux_dir = aux_output_dir_name(config, testfile);
            compose_and_run(config,
                            testfile,
                            make_run_args(config, props, testfile),
                            env,
                            config.run_lib_path.as_slice(),
                            Some(aux_dir.as_str().unwrap()),
                            None)
        }
    }
}

fn compose_and_run_compiler(
    config: &Config,
    props: &TestProps,
    testfile: &Path,
    args: ProcArgs,
    input: Option<String>) -> ProcRes {

    if !props.aux_builds.is_empty() {
        ensure_dir(&aux_output_dir_name(config, testfile));
    }

    let aux_dir = aux_output_dir_name(config, testfile);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let extra_link_args = vec!("-L".to_string(), aux_dir.as_str().unwrap().to_string());

    for rel_ab in props.aux_builds.iter() {
        let abs_ab = config.aux_base.join(rel_ab.as_slice());
        let aux_props = header::load_props(&abs_ab);
        let crate_type = if aux_props.no_prefer_dynamic {
            Vec::new()
        } else {
            vec!("--crate-type=dylib".to_string())
        };
        let aux_args =
            make_compile_args(config,
                              &aux_props,
                              crate_type.append(
                                  extra_link_args.as_slice()),
                              |a,b| {
                                  let f = make_lib_name(a, b, testfile);
                                  ThisDirectory(f.dir_path())
                              },
                              &abs_ab);
        let auxres = compose_and_run(config,
                                     &abs_ab,
                                     aux_args,
                                     Vec::new(),
                                     config.compile_lib_path.as_slice(),
                                     Some(aux_dir.as_str().unwrap()),
                                     None);
        if !auxres.status.success() {
            fatal_proc_rec(
                format!("auxiliary build of {} failed to compile: ",
                        abs_ab.display()).as_slice(),
                &auxres);
        }

        match config.target.as_slice() {
            "arm-linux-androideabi" => {
                _arm_push_aux_shared_library(config, testfile);
            }
            _ => {}
        }
    }

    compose_and_run(config,
                    testfile,
                    args,
                    Vec::new(),
                    config.compile_lib_path.as_slice(),
                    Some(aux_dir.as_str().unwrap()),
                    input)
}

fn ensure_dir(path: &Path) {
    if path.is_dir() { return; }
    fs::mkdir(path, io::UserRWX).unwrap();
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
    ThisFile(Path),
    ThisDirectory(Path),
}

fn make_compile_args(config: &Config,
                     props: &TestProps,
                     extras: Vec<String> ,
                     xform: |&Config, &Path| -> TargetLocation,
                     testfile: &Path)
                     -> ProcArgs {
    let xform_file = xform(config, testfile);
    let target = if props.force_host {
        config.host.as_slice()
    } else {
        config.target.as_slice()
    };
    // FIXME (#9639): This needs to handle non-utf8 paths
    let mut args = vec!(testfile.as_str().unwrap().to_string(),
                        "-L".to_string(),
                        config.build_base.as_str().unwrap().to_string(),
                        format!("--target={}", target));
    args.push_all(extras.as_slice());
    if !props.no_prefer_dynamic {
        args.push("-C".to_string());
        args.push("prefer-dynamic".to_string());
    }
    let path = match xform_file {
        ThisFile(path) => {
            args.push("-o".to_string());
            path
        }
        ThisDirectory(path) => {
            args.push("--out-dir".to_string());
            path
        }
    };
    args.push(path.as_str().unwrap().to_string());
    if props.force_host {
        args.push_all_move(split_maybe_args(&config.host_rustcflags));
    } else {
        args.push_all_move(split_maybe_args(&config.target_rustcflags));
    }
    args.push_all_move(split_maybe_args(&props.compile_flags));
    return ProcArgs {
        prog: config.rustc_path.as_str().unwrap().to_string(),
        args: args,
    };
}

fn make_lib_name(config: &Config, auxfile: &Path, testfile: &Path) -> Path {
    // what we return here is not particularly important, as it
    // happens; rustc ignores everything except for the directory.
    let auxname = output_testname(auxfile);
    aux_output_dir_name(config, testfile).join(&auxname)
}

fn make_exe_name(config: &Config, testfile: &Path) -> Path {
    let mut f = output_base_name(config, testfile);
    if !os::consts::EXE_SUFFIX.is_empty() {
        match f.filename().map(|s| Vec::from_slice(s).append(os::consts::EXE_SUFFIX.as_bytes())) {
            Some(v) => f.set_filename(v),
            None => ()
        }
    }
    f
}

fn make_run_args(config: &Config, props: &TestProps, testfile: &Path) ->
   ProcArgs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    let mut args = split_maybe_args(&config.runtool);
    let exe_file = make_exe_name(config, testfile);

    // FIXME (#9639): This needs to handle non-utf8 paths
    args.push(exe_file.as_str().unwrap().to_string());

    // Add the arguments in the run_flags directive
    args.push_all_move(split_maybe_args(&props.run_flags));

    let prog = args.remove(0).unwrap();
    return ProcArgs {
        prog: prog,
        args: args,
    };
}

fn split_maybe_args(argstr: &Option<String>) -> Vec<String> {
    match *argstr {
        Some(ref s) => {
            s.as_slice()
             .split(' ')
             .filter_map(|s| {
                 if s.is_whitespace() {
                     None
                 } else {
                     Some(s.to_string())
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
                                       prog.as_slice(),
                                       args.as_slice());
            logv(config, format!("executing {}", cmdline));
            cmdline
        };
    let procsrv::Result {
        out,
        err,
        status
    } = procsrv::run(lib_path,
                     prog.as_slice(),
                     aux_path,
                     args.as_slice(),
                     env,
                     input).expect(format!("failed to exec `{}`", prog).as_slice());
    dump_output(config, testfile, out.as_slice(), err.as_slice());
    return ProcRes {
        status: status,
        stdout: out,
        stderr: err,
        cmdline: cmdline,
    };
}

// Linux and mac don't require adjusting the library search path
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
#[cfg(target_os = "dragonfly")]
fn make_cmdline(_libpath: &str, prog: &str, args: &[String]) -> String {
    format!("{} {}", prog, args.connect(" "))
}

#[cfg(target_os = "windows")]
fn make_cmdline(libpath: &str, prog: &str, args: &[String]) -> String {
    format!("{} {} {}", lib_path_cmd_prefix(libpath), prog, args.connect(" "))
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
#[cfg(target_os = "windows")]
fn lib_path_cmd_prefix(path: &str) -> String {
    format!("{}=\"{}\"", util::lib_path_env_var(), util::make_new_path(path))
}

fn dump_output(config: &Config, testfile: &Path, out: &str, err: &str) {
    dump_output_file(config, testfile, out, "out");
    dump_output_file(config, testfile, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

fn dump_output_file(config: &Config, testfile: &Path,
                    out: &str, extension: &str) {
    let outfile = make_out_name(config, testfile, extension);
    File::create(&outfile).write(out.as_bytes()).unwrap();
}

fn make_out_name(config: &Config, testfile: &Path, extension: &str) -> Path {
    output_base_name(config, testfile).with_extension(extension)
}

fn aux_output_dir_name(config: &Config, testfile: &Path) -> Path {
    let mut f = output_base_name(config, testfile);
    match f.filename().map(|s| Vec::from_slice(s).append(b".libaux")) {
        Some(v) => f.set_filename(v),
        None => ()
    }
    f
}

fn output_testname(testfile: &Path) -> Path {
    Path::new(testfile.filestem().unwrap())
}

fn output_base_name(config: &Config, testfile: &Path) -> Path {
    config.build_base
        .join(&output_testname(testfile))
        .with_extension(config.stage_id.as_slice())
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

fn fatal(err: &str) -> ! { error(err); fail!(); }

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
    fail!();
}

fn _arm_exec_compiled_test(config: &Config,
                           props: &TestProps,
                           testfile: &Path,
                           env: Vec<(String, String)>)
                           -> ProcRes {
    let args = make_run_args(config, props, testfile);
    let cmdline = make_cmdline("",
                               args.prog.as_slice(),
                               args.args.as_slice());

    // get bare program string
    let mut tvec: Vec<String> = args.prog
                                    .as_slice()
                                    .split('/')
                                    .map(|ts| ts.to_string())
                                    .collect();
    let prog_short = tvec.pop().unwrap();

    // copy to target
    let copy_result = procsrv::run("",
                                   config.adb_path.as_slice(),
                                   None,
                                   [
                                    "push".to_string(),
                                    args.prog.clone(),
                                    config.adb_test_dir.clone()
                                   ],
                                   vec!(("".to_string(), "".to_string())),
                                   Some("".to_string()))
        .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

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
    runargs.push("shell".to_string());
    for (key, val) in env.into_iter() {
        runargs.push(format!("{}={}", key, val));
    }
    runargs.push(format!("{}/adb_run_wrapper.sh", config.adb_test_dir));
    runargs.push(format!("{}", config.adb_test_dir));
    runargs.push(format!("{}", prog_short));

    for tv in args.args.iter() {
        runargs.push(tv.to_string());
    }
    procsrv::run("",
                 config.adb_path.as_slice(),
                 None,
                 runargs.as_slice(),
                 vec!(("".to_string(), "".to_string())), Some("".to_string()))
        .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

    // get exitcode of result
    runargs = Vec::new();
    runargs.push("shell".to_string());
    runargs.push("cat".to_string());
    runargs.push(format!("{}/{}.exitcode", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: exitcode_out, err: _, status: _ } =
        procsrv::run("",
                     config.adb_path.as_slice(),
                     None,
                     runargs.as_slice(),
                     vec!(("".to_string(), "".to_string())),
                     Some("".to_string()))
        .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

    let mut exitcode: int = 0;
    for c in exitcode_out.as_slice().chars() {
        if !c.is_digit() { break; }
        exitcode = exitcode * 10 + match c {
            '0' .. '9' => c as int - ('0' as int),
            _ => 101,
        }
    }

    // get stdout of result
    runargs = Vec::new();
    runargs.push("shell".to_string());
    runargs.push("cat".to_string());
    runargs.push(format!("{}/{}.stdout", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stdout_out, err: _, status: _ } =
        procsrv::run("",
                     config.adb_path.as_slice(),
                     None,
                     runargs.as_slice(),
                     vec!(("".to_string(), "".to_string())),
                     Some("".to_string()))
        .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

    // get stderr of result
    runargs = Vec::new();
    runargs.push("shell".to_string());
    runargs.push("cat".to_string());
    runargs.push(format!("{}/{}.stderr", config.adb_test_dir, prog_short));

    let procsrv::Result{ out: stderr_out, err: _, status: _ } =
        procsrv::run("",
                     config.adb_path.as_slice(),
                     None,
                     runargs.as_slice(),
                     vec!(("".to_string(), "".to_string())),
                     Some("".to_string()))
        .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

    dump_output(config,
                testfile,
                stdout_out.as_slice(),
                stderr_out.as_slice());

    ProcRes {
        status: process::ExitStatus(exitcode),
        stdout: stdout_out,
        stderr: stderr_out,
        cmdline: cmdline
    }
}

fn _arm_push_aux_shared_library(config: &Config, testfile: &Path) {
    let tdir = aux_output_dir_name(config, testfile);

    let dirs = fs::readdir(&tdir).unwrap();
    for file in dirs.iter() {
        if file.extension_str() == Some("so") {
            // FIXME (#9639): This needs to handle non-utf8 paths
            let copy_result = procsrv::run("",
                                           config.adb_path.as_slice(),
                                           None,
                                           [
                                            "push".to_string(),
                                            file.as_str()
                                                .unwrap()
                                                .to_string(),
                                            config.adb_test_dir.to_string()
                                           ],
                                           vec!(("".to_string(),
                                                 "".to_string())),
                                           Some("".to_string()))
                .expect(format!("failed to exec `{}`", config.adb_path).as_slice());

            if config.verbose {
                println!("push ({}) {} {} {}",
                    config.target, file.display(),
                    copy_result.out, copy_result.err);
            }
        }
    }
}

// codegen tests (vs. clang)

fn append_suffix_to_stem(p: &Path, suffix: &str) -> Path {
    if suffix.len() == 0 {
        (*p).clone()
    } else {
        let stem = p.filestem().unwrap();
        p.with_filename(Vec::from_slice(stem).append(b"-").append(suffix.as_bytes()))
    }
}

fn compile_test_and_save_bitcode(config: &Config, props: &TestProps,
                                 testfile: &Path) -> ProcRes {
    let aux_dir = aux_output_dir_name(config, testfile);
    // FIXME (#9639): This needs to handle non-utf8 paths
    let link_args = vec!("-L".to_string(),
                         aux_dir.as_str().unwrap().to_string());
    let llvm_args = vec!("--emit=bc,obj".to_string(),
                         "--crate-type=lib".to_string());
    let args = make_compile_args(config,
                                 props,
                                 link_args.append(llvm_args.as_slice()),
                                 |a, b| ThisDirectory(output_base_name(a, b).dir_path()),
                                 testfile);
    compose_and_run_compiler(config, props, testfile, args, None)
}

fn compile_cc_with_clang_and_save_bitcode(config: &Config, _props: &TestProps,
                                          testfile: &Path) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_extension("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, "clang");
    let testcc = testfile.with_extension("cc");
    let proc_args = ProcArgs {
        // FIXME (#9639): This needs to handle non-utf8 paths
        prog: config.clang_path.as_ref().unwrap().as_str().unwrap().to_string(),
        args: vec!("-c".to_string(),
                   "-emit-llvm".to_string(),
                   "-o".to_string(),
                   bitcodefile.as_str().unwrap().to_string(),
                   testcc.as_str().unwrap().to_string())
    };
    compose_and_run(config, testfile, proc_args, Vec::new(), "", None, None)
}

fn extract_function_from_bitcode(config: &Config, _props: &TestProps,
                                 fname: &str, testfile: &Path,
                                 suffix: &str) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_extension("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, suffix);
    let extracted_bc = append_suffix_to_stem(&bitcodefile, "extract");
    let prog = config.llvm_bin_path.as_ref().unwrap().join("llvm-extract");
    let proc_args = ProcArgs {
        // FIXME (#9639): This needs to handle non-utf8 paths
        prog: prog.as_str().unwrap().to_string(),
        args: vec!(format!("-func={}", fname),
                   format!("-o={}", extracted_bc.as_str().unwrap()),
                   bitcodefile.as_str().unwrap().to_string())
    };
    compose_and_run(config, testfile, proc_args, Vec::new(), "", None, None)
}

fn disassemble_extract(config: &Config, _props: &TestProps,
                       testfile: &Path, suffix: &str) -> ProcRes {
    let bitcodefile = output_base_name(config, testfile).with_extension("bc");
    let bitcodefile = append_suffix_to_stem(&bitcodefile, suffix);
    let extracted_bc = append_suffix_to_stem(&bitcodefile, "extract");
    let extracted_ll = extracted_bc.with_extension("ll");
    let prog = config.llvm_bin_path.as_ref().unwrap().join("llvm-dis");
    let proc_args = ProcArgs {
        // FIXME (#9639): This needs to handle non-utf8 paths
        prog: prog.as_str().unwrap().to_string(),
        args: vec!(format!("-o={}", extracted_ll.as_str().unwrap()),
                   extracted_bc.as_str().unwrap().to_string())
    };
    compose_and_run(config, testfile, proc_args, Vec::new(), "", None, None)
}


fn count_extracted_lines(p: &Path) -> uint {
    let x = File::open(&p.with_extension("ll")).read_to_end().unwrap();
    let x = str::from_utf8(x.as_slice()).unwrap();
    x.lines().count()
}


fn run_codegen_test(config: &Config, props: &TestProps,
                    testfile: &Path, mm: &mut MetricMap) {

    if config.llvm_bin_path.is_none() {
        fatal("missing --llvm-bin-path");
    }

    if config.clang_path.is_none() {
        fatal("missing --clang-path");
    }

    let mut proc_res = compile_test_and_save_bitcode(config, props, testfile);
    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    proc_res = extract_function_from_bitcode(config, props, "test", testfile, "");
    if !proc_res.status.success() {
        fatal_proc_rec("extracting 'test' function failed",
                      &proc_res);
    }

    proc_res = disassemble_extract(config, props, testfile, "");
    if !proc_res.status.success() {
        fatal_proc_rec("disassembling extract failed", &proc_res);
    }


    let mut proc_res = compile_cc_with_clang_and_save_bitcode(config, props, testfile);
    if !proc_res.status.success() {
        fatal_proc_rec("compilation failed!", &proc_res);
    }

    proc_res = extract_function_from_bitcode(config, props, "test", testfile, "clang");
    if !proc_res.status.success() {
        fatal_proc_rec("extracting 'test' function failed",
                      &proc_res);
    }

    proc_res = disassemble_extract(config, props, testfile, "clang");
    if !proc_res.status.success() {
        fatal_proc_rec("disassembling extract failed", &proc_res);
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
