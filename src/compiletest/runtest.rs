import std::io;
import io::writer_util;
import std::fs;
import std::os;

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

fn run(cx: cx, -_testfile: [u8]) {
    let testfile = str::unsafe_from_bytes(_testfile);
    if cx.config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        io::stdout().write_str("\n\n");
    }
    #debug("running %s", testfile);
    let props = load_props(testfile);
    alt cx.config.mode {
      mode_compile_fail. { run_cfail_test(cx, props, testfile); }
      mode_run_fail. { run_rfail_test(cx, props, testfile); }
      mode_run_pass. { run_rpass_test(cx, props, testfile); }
      mode_pretty. { run_pretty_test(cx, props, testfile); }
    }
}

fn run_cfail_test(cx: cx, props: test_props, testfile: str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status == 0 {
        fatal_procres("compile-fail test compiled successfully!", procres);
    }

    check_correct_failure_status(procres);

    let expected_errors = errors::load_errors(testfile);
    if vec::is_not_empty(expected_errors) {
        if vec::is_not_empty(props.error_patterns) {
            fatal("both error pattern and expected errors specified");
        }
        check_expected_errors(expected_errors, testfile, procres);
    } else {
        check_error_patterns(props, testfile, procres);
    }
}

fn run_rfail_test(cx: cx, props: test_props, testfile: str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, props, testfile);

    // The value our Makefile configures valgrind to return on failure
    const valgrind_err: int = 100;
    if procres.status == valgrind_err {
        fatal_procres("run-fail test isn't valgrind-clean!", procres);
    }

    check_correct_failure_status(procres);
    check_error_patterns(props, testfile, procres);
}

fn check_correct_failure_status(procres: procres) {
    // The value the rust runtime returns on failure
    const rust_err: int = 101;
    if procres.status != rust_err {
        fatal_procres(
            #fmt("failure produced the wrong error code: %d",
                 procres.status),
            procres);
    }
}

fn run_rpass_test(cx: cx, props: test_props, testfile: str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, props, testfile);


    if procres.status != 0 { fatal_procres("test run failed!", procres); }
}

fn run_pretty_test(cx: cx, props: test_props, testfile: str) {
    if option::is_some(props.pp_exact) {
        logv(cx.config, "testing for exact pretty-printing");
    } else { logv(cx.config, "testing for converging pretty-printing"); }

    let rounds =
        alt props.pp_exact { option::some(_) { 1 } option::none. { 2 } };

    let srcs = [result::get(io::read_whole_file_str(testfile))];

    let round = 0;
    while round < rounds {
        logv(cx.config, #fmt["pretty-printing round %d", round]);
        let procres = print_source(cx, testfile, srcs[round]);

        if procres.status != 0 {
            fatal_procres(#fmt["pretty-printing failed in round %d", round],
                          procres);
        }

        srcs += [procres.stdout];
        round += 1;
    }

    let expected =
        alt props.pp_exact {
          option::some(file) {
            let filepath = fs::connect(fs::dirname(testfile), file);
            result::get(io::read_whole_file_str(filepath))
          }
          option::none. { srcs[vec::len(srcs) - 2u] }
        };
    let actual = srcs[vec::len(srcs) - 1u];

    if option::is_some(props.pp_exact) {
        // Now we have to care about line endings
        let cr = "\r";
        check (str::is_not_empty(cr));
        actual = str::replace(actual, cr, "");
        expected = str::replace(expected, cr, "");
    }

    compare_source(expected, actual);

    // Finally, let's make sure it actually appears to remain valid code
    let procres = typecheck_source(cx, testfile, actual);

    if procres.status != 0 {
        fatal_procres("pretty-printed source does not typecheck", procres);
    }

    ret;

    fn print_source(cx: cx, testfile: str, src: str) -> procres {
        compose_and_run(cx, testfile, make_pp_args,
                        cx.config.compile_lib_path, option::some(src))
    }

    fn make_pp_args(config: config, _testfile: str) -> procargs {
        let prog = config.rustc_path;
        let args = ["-", "--pretty", "normal"];
        ret {prog: prog, args: args};
    }

    fn compare_source(expected: str, actual: str) {
        if expected != actual {
            error("pretty-printed source does not match expected source");
            let msg =
                #fmt["\n\
expected:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
actual:\n\
------------------------------------------\n\
%s\n\
------------------------------------------\n\
\n",
                     expected, actual];
            io::stdout().write_str(msg);
            fail;
        }
    }

    fn typecheck_source(cx: cx, testfile: str, src: str) -> procres {
        compose_and_run(cx, testfile, make_typecheck_args,
                        cx.config.compile_lib_path, option::some(src))
    }

    fn make_typecheck_args(config: config, _testfile: str) -> procargs {
        let prog = config.rustc_path;
        let args = ["-", "--no-trans", "--lib"];
        args += split_maybe_args(config.rustcflags);
        ret {prog: prog, args: args};
    }
}

fn check_error_patterns(props: test_props,
                        testfile: str,
                        procres: procres) {
    if vec::is_empty(props.error_patterns) {
        fatal("no error pattern specified in " + testfile);
    }

    if procres.status == 0 {
        fatal("process did not return an error status");
    }

    let next_err_idx = 0u;
    let next_err_pat = props.error_patterns[next_err_idx];
    for line: str in str::split(procres.stdout, '\n' as u8) {
        if str::find(line, next_err_pat) > 0 {
            #debug("found error pattern %s", next_err_pat);
            next_err_idx += 1u;
            if next_err_idx == vec::len(props.error_patterns) {
                #debug("found all error patterns");
                ret;
            }
            next_err_pat = props.error_patterns[next_err_idx];
        }
    }

    let missing_patterns =
        vec::slice(props.error_patterns, next_err_idx,
                   vec::len(props.error_patterns));
    if vec::len(missing_patterns) == 1u {
        fatal_procres(#fmt["error pattern '%s' not found!",
                           missing_patterns[0]], procres);
    } else {
        for pattern: str in missing_patterns {
            error(#fmt["error pattern '%s' not found!", pattern]);
        }
        fatal_procres("multiple error patterns not found", procres);
    }
}

fn check_expected_errors(expected_errors: [errors::expected_error],
                         testfile: str,
                         procres: procres) {

    // true if we found the error in question
    let found_flags = vec::init_elt_mut(false, vec::len(expected_errors));

    if procres.status == 0 {
        fatal("process did not return an error status");
    }

    let prefixes = vec::map(expected_errors, {|ee|
        #fmt("%s:%u:", testfile, ee.line)
    });

    // Scan and extract our error/warning messages,
    // which look like:
    //    filename:line1:col1: line2:col2: *error:* msg
    //    filename:line1:col1: line2:col2: *warning:* msg
    // where line1:col1: is the starting point, line2:col2:
    // is the ending point, and * represents ANSI color codes.
    for line: str in str::split(procres.stdout, '\n' as u8) {
        let was_expected = false;
        vec::iteri(expected_errors) {|i, ee|
            if !found_flags[i] {
                #debug["prefix=%s ee.kind=%s ee.msg=%s line=%s",
                       prefixes[i], ee.kind, ee.msg, line];
                if (str::starts_with(line, prefixes[i]) &&
                    str::contains(line, ee.kind) &&
                    str::contains(line, ee.msg)) {
                    found_flags[i] = true;
                    was_expected = true;
                }
            }
        }

        // ignore this msg which gets printed at the end
        if str::contains(line, "aborting due to previous errors") {
            was_expected = true;
        }

        if !was_expected && (str::contains(line, "error") ||
                             str::contains(line, "warning")) {
            fatal_procres(#fmt["unexpected error pattern '%s'!", line],
                          procres);
        }
    }

    uint::range(0u, vec::len(found_flags)) {|i|
        if !found_flags[i] {
            let ee = expected_errors[i];
            fatal_procres(#fmt["expected %s on line %u not found: %s",
                               ee.kind, ee.line, ee.msg], procres);
        }
    }
}

type procargs = {prog: str, args: [str]};

type procres = {status: int, stdout: str, stderr: str, cmdline: str};

fn compile_test(cx: cx, props: test_props, testfile: str) -> procres {
    compose_and_run(cx, testfile, bind make_compile_args(_, props, _),
                    cx.config.compile_lib_path, option::none)
}

fn exec_compiled_test(cx: cx, props: test_props, testfile: str) -> procres {
    compose_and_run(cx, testfile, bind make_run_args(_, props, _),
                    cx.config.run_lib_path, option::none)
}

fn compose_and_run(cx: cx, testfile: str,
                   make_args: fn@(config, str) -> procargs, lib_path: str,
                   input: option::t<str>) -> procres {
    let procargs = make_args(cx.config, testfile);
    ret program_output(cx, testfile, lib_path, procargs.prog, procargs.args,
                       input);
}

fn make_compile_args(config: config, props: test_props, testfile: str) ->
   procargs {
    let prog = config.rustc_path;
    let args = [testfile, "-o", make_exe_name(config, testfile)];
    args += split_maybe_args(config.rustcflags);
    args += split_maybe_args(props.compile_flags);
    ret {prog: prog, args: args};
}

fn make_exe_name(config: config, testfile: str) -> str {
    output_base_name(config, testfile) + os::exec_suffix()
}

fn make_run_args(config: config, _props: test_props, testfile: str) ->
   procargs {
    let toolargs = {
            // If we've got another tool to run under (valgrind),
            // then split apart its command
            let runtool =
                alt config.runtool {
                  option::some(s) { option::some(s) }
                  option::none. { option::none }
                };
            split_maybe_args(runtool)
        };

    let args = toolargs + [make_exe_name(config, testfile)];
    ret {prog: args[0], args: vec::slice(args, 1u, vec::len(args))};
}

fn split_maybe_args(argstr: option::t<str>) -> [str] {
    fn rm_whitespace(v: [str]) -> [str] {
        fn flt(&&s: str) -> option::t<str> {
            if !is_whitespace(s) { option::some(s) } else { option::none }
        }

        // FIXME: This should be in std
        fn is_whitespace(s: str) -> bool {
            for c: u8 in s { if c != ' ' as u8 { ret false; } }
            ret true;
        }
        vec::filter_map(v, flt)
    }

    alt argstr {
      option::some(s) { rm_whitespace(str::split(s, ' ' as u8)) }
      option::none. { [] }
    }
}

fn program_output(cx: cx, testfile: str, lib_path: str, prog: str,
                  args: [str], input: option::t<str>) -> procres {
    let cmdline =
        {
            let cmdline = make_cmdline(lib_path, prog, args);
            logv(cx.config, #fmt["executing %s", cmdline]);
            cmdline
        };
    let res = procsrv::run(cx.procsrv, lib_path, prog, args, input);
    dump_output(cx.config, testfile, res.out, res.err);
    ret {status: res.status,
         stdout: res.out,
         stderr: res.err,
         cmdline: cmdline};
}

// Linux and mac don't require adjusting the library search path
#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn make_cmdline(_libpath: str, prog: str, args: [str]) -> str {
    #fmt["%s %s", prog, str::connect(args, " ")]
}

#[cfg(target_os = "win32")]
fn make_cmdline(libpath: str, prog: str, args: [str]) -> str {
    #fmt["%s %s %s", lib_path_cmd_prefix(libpath), prog,
         str::connect(args, " ")]
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
fn lib_path_cmd_prefix(path: str) -> str {
    #fmt["%s=\"%s\"", util::lib_path_env_var(), util::make_new_path(path)]
}

fn dump_output(config: config, testfile: str, out: str, err: str) {
    dump_output_file(config, testfile, out, "out");
    dump_output_file(config, testfile, err, "err");
    maybe_dump_to_stdout(config, out, err);
}

fn dump_output_file(config: config, testfile: str, out: str, extension: str) {
    let outfile = make_out_name(config, testfile, extension);
    let writer = result::get(
        io::file_writer(outfile, [io::create, io::truncate]));
    writer.write_str(out);
}

fn make_out_name(config: config, testfile: str, extension: str) -> str {
    output_base_name(config, testfile) + "." + extension
}

fn output_base_name(config: config, testfile: str) -> str {
    let base = config.build_base;
    let filename =
        {
            let parts = str::split(fs::basename(testfile), '.' as u8);
            parts = vec::slice(parts, 0u, vec::len(parts) - 1u);
            str::connect(parts, ".")
        };
    #fmt["%s%s.%s", base, filename, config.stage_id]
}

fn maybe_dump_to_stdout(config: config, out: str, err: str) {
    if config.verbose {
        let sep1 = #fmt["------%s------------------------------", "stdout"];
        let sep2 = #fmt["------%s------------------------------", "stderr"];
        let sep3 = "------------------------------------------";
        io::stdout().write_line(sep1);
        io::stdout().write_line(out);
        io::stdout().write_line(sep2);
        io::stdout().write_line(err);
        io::stdout().write_line(sep3);
    }
}

fn error(err: str) { io::stdout().write_line(#fmt["\nerror: %s", err]); }

fn fatal(err: str) -> ! { error(err); fail; }

fn fatal_procres(err: str, procres: procres) -> ! {
    let msg =
        #fmt["\n\
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
             err, procres.cmdline, procres.stdout, procres.stderr];
    io::stdout().write_str(msg);
    fail;
}
