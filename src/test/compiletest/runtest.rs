import std::io;
import std::str;
import std::option;
import std::fs;
import std::os;
import std::ivec;
import std::test;

import common::mode_run_pass;
import common::mode_run_fail;
import common::mode_compile_fail;
import common::mode_pretty;
import common::cx;
import common::config;
import header::load_props;
import header::test_props;
import util::logv;

export run;

fn run(cx: &cx, _testfile: -[u8]) {
    let testfile = str::unsafe_from_bytes(_testfile);
    test::configure_test_task();
    if (cx.config.verbose) {
        // We're going to be dumping a lot of info. Start on a new line.
        io::stdout().write_str("\n\n");
    }
    log #fmt("running %s", testfile);
    let props = load_props(testfile);
    alt cx.config.mode {
      mode_compile_fail. { run_cfail_test(cx, props, testfile); }
      mode_run_fail. { run_rfail_test(cx, props, testfile); }
      mode_run_pass. { run_rpass_test(cx, props, testfile); }
      mode_pretty. { run_pretty_test(cx, props, testfile); }
    }
}

fn run_cfail_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status == 0 {
        fatal_procres("compile-fail test compiled successfully!",
                      procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rfail_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 {
        fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, props, testfile);

    if procres.status == 0 {
        fatal_procres("run-fail test didn't produce an error!",
                      procres);
    }

    // This is the value valgrind returns on failure
    // FIXME: Why is this value neither the value we pass to
    // valgrind as --error-exitcode (1), nor the value we see as the
    // exit code on the command-line (137)?
    const valgrind_err: int = 9;
    if procres.status == valgrind_err {
        fatal_procres("run-fail test isn't valgrind-clean!", procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rpass_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 {
        fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, props, testfile);


    if procres.status != 0 { fatal_procres("test run failed!", procres); }
}

fn run_pretty_test(cx: &cx, props: &test_props, testfile: &str) {
    if option::is_some(props.pp_exact) {
        logv(cx.config, "testing for exact pretty-printing");
    } else {
        logv(cx.config, "testing for converging pretty-printing");
    }

    let rounds = alt props.pp_exact {
      option::some(_) { 1 }
      option::none. { 2 }
    };

    let srcs = ~[io::read_whole_file_str(testfile)];

    let round = 0;
    while round < rounds {
        logv(cx.config, #fmt("pretty-printing round %d", round));
        let procres = print_source(cx, testfile, srcs.(round));

        if procres.status != 0 {
            fatal_procres(#fmt("pretty-printing failed in round %d", round),
                          procres);
        }

        srcs += ~[procres.stdout];
        round += 1;
    }

    let expected = alt props.pp_exact {
      option::some(file) {
        let filepath = fs::connect(fs::dirname(testfile), file);
        io::read_whole_file_str(filepath)
      }
      option::none. {
        srcs.(ivec::len(srcs) - 2u)
      }
    };
    let actual = srcs.(ivec::len(srcs) - 1u);

    if option::is_some(props.pp_exact) {
        // Now we have to care about line endings
        let cr = "\r";
        check str::is_not_empty(cr);
        actual = str::replace(actual, cr, "");
        expected = str::replace(expected, cr, "");
    }

    compare_source(expected, actual);

    // Finally, let's make sure it actually appears to remain valid code
    let procres = typecheck_source(cx, testfile, actual);

    if procres.status != 0 {
        fatal_procres("pretty-printed source does not typecheck",
                      procres);
    }

    ret;

    fn print_source(cx: &cx, testfile: &str, src: &str) -> procres {
        compose_and_run(cx, testfile, make_pp_args,
                        cx.config.compile_lib_path, option::some(src))
    }

    fn make_pp_args(config: &config, testfile: &str) -> procargs {
        let prog = config.rustc_path;
        let args = ~["-", "--pretty", "normal"];
        ret {prog: prog, args: args};
    }

    fn compare_source(expected: &str, actual: &str) {
        if expected != actual {
            error("pretty-printed source does match expected source");
            let msg = #fmt("\n\
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
            fail;
        }
    }

    fn typecheck_source(cx: &cx, testfile: &str, src: &str) -> procres {
        compose_and_run(cx, testfile, make_typecheck_args,
                        cx.config.compile_lib_path, option::some(src))
    }

    fn make_typecheck_args(config: &config, testfile: &str) -> procargs {
        let prog = config.rustc_path;
        let args = ~["-", "--no-trans", "--lib"];
        ret {prog: prog, args: args};
    }
}

fn check_error_patterns(props: &test_props, testfile: &str,
                        procres: &procres) {
    if ivec::is_empty(props.error_patterns) {
        fatal("no error pattern specified in " + testfile);
    }

    let next_err_idx = 0u;
    let next_err_pat = props.error_patterns.(next_err_idx);
    for line: str in str::split(procres.stdout, '\n' as u8) {
        if str::find(line, next_err_pat) > 0 {
            log #fmt("found error pattern %s", next_err_pat);
            next_err_idx += 1u;
            if next_err_idx == ivec::len(props.error_patterns) {
                log "found all error patterns";
                ret;
            }
            next_err_pat = props.error_patterns.(next_err_idx);
        }
    }

    let missing_patterns =
        ivec::slice(props.error_patterns, next_err_idx,
                    ivec::len(props.error_patterns));
    if ivec::len(missing_patterns) == 1u {
        fatal_procres(#fmt("error pattern '%s' not found!",
                           missing_patterns.(0)), procres);
    } else {
        for pattern: str in missing_patterns {
            error(#fmt("error pattern '%s' not found!", pattern));
        }
        fatal_procres("multiple error patterns not found", procres);
    }
}

type procargs = {prog: str, args: [str]};

type procres = {status: int, stdout: str, stderr: str, cmdline: str};

fn compile_test(cx: &cx, props: &test_props, testfile: &str) -> procres {
    compose_and_run(cx, testfile, bind make_compile_args(_, props, _),
                    cx.config.compile_lib_path, option::none)
}

fn exec_compiled_test(cx: &cx, props: &test_props,
                      testfile: &str) -> procres {
    compose_and_run(cx, testfile, bind make_run_args(_, props, _),
                    cx.config.run_lib_path, option::none)
}

fn compose_and_run(cx: &cx, testfile: &str,
                   make_args: fn(&config, &str) -> procargs ,
                   lib_path: &str,
                   input: option::t[str]) -> procres {
    let procargs = make_args(cx.config, testfile);
    ret program_output(cx, testfile, lib_path,
                       procargs.prog, procargs.args,
                       input);
}

fn make_compile_args(config: &config,
                     props: &test_props, testfile: &str) ->
    procargs {
    let prog = config.rustc_path;
    let args = ~[testfile, "-o", make_exe_name(config, testfile)];
    args += split_maybe_args(config.rustcflags);
    args += split_maybe_args(props.compile_flags);
    ret {prog: prog, args: args};
}

fn make_exe_name(config: &config, testfile: &str) -> str {
    output_base_name(config, testfile) + os::exec_suffix()
}

fn make_run_args(config: &config,
                 props: &test_props, testfile: &str) -> procargs {
    let toolargs = if !props.no_valgrind {
        // If we've got another tool to run under (valgrind),
        // then split apart its command
        split_maybe_args(config.runtool)
    } else {
        ~[]
    };

    let args = toolargs + ~[make_exe_name(config, testfile)];
    ret {prog: args.(0), args: ivec::slice(args, 1u, ivec::len(args))};
}

fn split_maybe_args(argstr: &option::t[str]) -> [str] {
    fn rm_whitespace(v: &[str]) -> [str] {
        fn flt(s: &str) -> option::t[str] {
            if !is_whitespace(s) {
                option::some(s)
            } else {
                option::none
            }
        }

        // FIXME: This should be in std
        fn is_whitespace(s: str) -> bool {
            for c: u8 in s {
                if c != (' ' as u8) { ret false; }
            }
            ret true;
        }
        ivec::filter_map(flt, v)
    }

    alt argstr {
      option::some(s) { rm_whitespace(str::split(s, ' ' as u8)) }
      option::none. { ~[] }
    }
}

fn program_output(cx: &cx, testfile: &str, lib_path: &str, prog: &str,
                  args: &[str], input: option::t[str]) -> procres {
    let cmdline =
    {
        let cmdline = make_cmdline(lib_path, prog, args);
        logv(cx.config, #fmt("executing %s", cmdline));
        cmdline
    };
    let res = procsrv::run(cx.procsrv, lib_path,
                           prog, args, input);
    dump_output(cx.config, testfile, res.out, res.err);
    ret {status: res.status, stdout: res.out,
         stderr: res.err, cmdline: cmdline};
}

fn make_cmdline(libpath: &str, prog: &str, args: &[str]) -> str {
    #fmt("%s %s %s", lib_path_cmd_prefix(libpath), prog,
         str::connect(args, " "))
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
fn lib_path_cmd_prefix(path: &str) -> str {
    #fmt("%s=\"%s\"", util::lib_path_env_var(), util::make_new_path(path))
}

fn dump_output(config: &config, testfile: &str,
               out: &str, err: &str) {
    dump_output_file(config, testfile, out, "out");
    dump_output_file(config, testfile, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

#[cfg(target_os = "win32")]
#[cfg(target_os = "linux")]
fn dump_output_file(config: &config, testfile: &str,
                    out: &str, extension: &str) {
    let outfile = make_out_name(config, testfile, extension);
    let writer = io::file_writer(outfile,
                                     ~[io::create, io::truncate]);
    writer.write_str(out);
}

// FIXME (726): Can't use file_writer on mac
#[cfg(target_os = "macos")]
fn dump_output_file(config: &config, testfile: &str,
                    out: &str, extension: &str) {
}

fn make_out_name(config: &config, testfile: &str,
                 extension: &str) -> str {
    output_base_name(config, testfile) + "." + extension
}

fn output_base_name(config: &config, testfile: &str) -> str {
    let base = config.build_base;
    let filename =
        {
            let parts = str::split(fs::basename(testfile), '.' as u8);
            parts = ivec::slice(parts, 0u, ivec::len(parts) - 1u);
            str::connect(parts, ".")
        };
    #fmt("%s%s.%s", base, filename, config.stage_id)
}

fn maybe_dump_to_stdout(config: &config,
                        out: &str, err: &str) {
    if config.verbose {
        let sep1 = #fmt("------%s------------------------------",
                        "stdout");
        let sep2 = #fmt("------%s------------------------------",
                        "stderr");
        let sep3 = "------------------------------------------";
        io::stdout().write_line(sep1);
        io::stdout().write_line(out);
        io::stdout().write_line(sep2);
        io::stdout().write_line(err);
        io::stdout().write_line(sep3);
    }
}

fn error(err: &str) { io::stdout().write_line(#fmt("\nerror: %s", err)); }

fn fatal(err: &str) -> ! { error(err); fail; }

fn fatal_procres(err: &str, procres: procres) -> ! {
    let msg =
        #fmt("\n\
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
             err, procres.cmdline, procres.stdout, procres.stderr);
    io::stdout().write_str(msg);
    fail;
}
