import io::writer_util;

import common::mode_run_pass;
import common::mode_run_fail;
import common::mode_compile_fail;
import common::mode_pretty;
import common::config;
import header::load_props;
import header::test_props;
import util::logv;

export run;

fn run(config: config, testfile: str) {
    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        io::stdout().write_str("\n\n");
    }
    #debug("running %s", testfile);
    let props = load_props(testfile);
    alt config.mode {
      mode_compile_fail { run_cfail_test(config, props, testfile); }
      mode_run_fail { run_rfail_test(config, props, testfile); }
      mode_run_pass { run_rpass_test(config, props, testfile); }
      mode_pretty { run_pretty_test(config, props, testfile); }
    }
}

fn run_cfail_test(config: config, props: test_props, testfile: str) {
    let procres = compile_test(config, props, testfile);

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

fn run_rfail_test(config: config, props: test_props, testfile: str) {
    let mut procres = compile_test(config, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(config, props, testfile);

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

fn run_rpass_test(config: config, props: test_props, testfile: str) {
    let mut procres = compile_test(config, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(config, props, testfile);

    if procres.status != 0 { fatal_procres("test run failed!", procres); }
}

fn run_pretty_test(config: config, props: test_props, testfile: str) {
    if option::is_some(props.pp_exact) {
        logv(config, "testing for exact pretty-printing");
    } else { logv(config, "testing for converging pretty-printing"); }

    let rounds =
        alt props.pp_exact { option::some(_) { 1 } option::none { 2 } };

    let mut srcs = [result::get(io::read_whole_file_str(testfile))];

    let mut round = 0;
    while round < rounds {
        logv(config, #fmt["pretty-printing round %d", round]);
        let procres = print_source(config, testfile, srcs[round]);

        if procres.status != 0 {
            fatal_procres(#fmt["pretty-printing failed in round %d", round],
                          procres);
        }

        srcs += [procres.stdout];
        round += 1;
    }

    let mut expected =
        alt props.pp_exact {
          option::some(file) {
            let filepath = path::connect(path::dirname(testfile), file);
            result::get(io::read_whole_file_str(filepath))
          }
          option::none { srcs[vec::len(srcs) - 2u] }
        };
    let mut actual = srcs[vec::len(srcs) - 1u];

    if option::is_some(props.pp_exact) {
        // Now we have to care about line endings
        let cr = "\r";
        check (str::is_not_empty(cr));
        actual = str::replace(actual, cr, "");
        expected = str::replace(expected, cr, "");
    }

    compare_source(expected, actual);

    // Finally, let's make sure it actually appears to remain valid code
    let procres = typecheck_source(config, props, testfile, actual);

    if procres.status != 0 {
        fatal_procres("pretty-printed source does not typecheck", procres);
    }

    ret;

    fn print_source(config: config, testfile: str, src: str) -> procres {
        compose_and_run(config, testfile, make_pp_args(config, testfile),
                        [], config.compile_lib_path, option::some(src))
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

    fn typecheck_source(config: config, props: test_props,
                        testfile: str, src: str) -> procres {
        compose_and_run_compiler(
            config, props, testfile,
            make_typecheck_args(config, testfile),
            option::some(src))
    }

    fn make_typecheck_args(config: config, testfile: str) -> procargs {
        let prog = config.rustc_path;
        let mut args = ["-", "--no-trans", "--lib", "-L", config.build_base,
                        "-L", aux_output_dir_name(config, testfile)];
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

    let mut next_err_idx = 0u;
    let mut next_err_pat = props.error_patterns[next_err_idx];
    let mut done = false;
    for str::split_char(procres.stderr, '\n').each {|line|
        if str::contains(line, next_err_pat) {
            #debug("found error pattern %s", next_err_pat);
            next_err_idx += 1u;
            if next_err_idx == vec::len(props.error_patterns) {
                #debug("found all error patterns");
                done = true;
                break;
            }
            next_err_pat = props.error_patterns[next_err_idx];
        }
    }
    if done { ret; }

    let missing_patterns =
        vec::slice(props.error_patterns, next_err_idx,
                   vec::len(props.error_patterns));
    if vec::len(missing_patterns) == 1u {
        fatal_procres(#fmt["error pattern '%s' not found!",
                           missing_patterns[0]], procres);
    } else {
        for missing_patterns.each {|pattern|
            error(#fmt["error pattern '%s' not found!", pattern]);
        }
        fatal_procres("multiple error patterns not found", procres);
    }
}

fn check_expected_errors(expected_errors: [errors::expected_error],
                         testfile: str,
                         procres: procres) {

    // true if we found the error in question
    let found_flags = vec::to_mut(vec::from_elem(
        vec::len(expected_errors), false));

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
    for str::split_char(procres.stderr, '\n').each {|line|
        let mut was_expected = false;
        for vec::eachi(expected_errors) {|i, ee|
            if !found_flags[i] {
                #debug["prefix=%s ee.kind=%s ee.msg=%s line=%s",
                       prefixes[i], ee.kind, ee.msg, line];
                if (str::starts_with(line, prefixes[i]) &&
                    str::contains(line, ee.kind) &&
                    str::contains(line, ee.msg)) {
                    found_flags[i] = true;
                    was_expected = true;
                    break;
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

    for uint::range(0u, vec::len(found_flags)) {|i|
        if !found_flags[i] {
            let ee = expected_errors[i];
            fatal_procres(#fmt["expected %s on line %u not found: %s",
                               ee.kind, ee.line, ee.msg], procres);
        }
    }
}

type procargs = {prog: str, args: [str]};

type procres = {status: int, stdout: str, stderr: str, cmdline: str};

fn compile_test(config: config, props: test_props,
                testfile: str) -> procres {
    let link_args = ["-L", aux_output_dir_name(config, testfile)];
    compose_and_run_compiler(
        config, props, testfile,
        make_compile_args(config, props, link_args,
                          make_exe_name, testfile),
        none)
}

fn exec_compiled_test(config: config, props: test_props,
                      testfile: str) -> procres {
    compose_and_run(config, testfile,
                    make_run_args(config, props, testfile),
                    props.exec_env,
                    config.run_lib_path, option::none)
}

fn compose_and_run_compiler(
    config: config,
    props: test_props,
    testfile: str,
    args: procargs,
    input: option<str>) -> procres {

    if props.aux_builds.is_not_empty() {
        ensure_dir(aux_output_dir_name(config, testfile));
    }

    let extra_link_args = ["-L", aux_output_dir_name(config, testfile)];

    vec::iter(props.aux_builds) {|rel_ab|
        let abs_ab = path::connect(config.aux_base, rel_ab);
        let aux_args =
            make_compile_args(config, props, ["--lib"] + extra_link_args,
                              bind make_lib_name(_, _, testfile), abs_ab);
        let auxres = compose_and_run(config, abs_ab, aux_args, [],
                                     config.compile_lib_path, option::none);
        if auxres.status != 0 {
            fatal_procres(
                #fmt["auxiliary build of %s failed to compile: ", abs_ab],
                auxres);
        }
    }

    compose_and_run(config, testfile, args, [],
                    config.compile_lib_path, input)
}

fn ensure_dir(path: path) {
    if os::path_is_dir(path) { ret; }
    if !os::make_dir(path, 0x1c0i32) {
        fail #fmt("can't make dir %s", path);
    }
}

fn compose_and_run(config: config, testfile: str,
                   procargs: procargs,
                   procenv: [(str, str)],
                   lib_path: str,
                   input: option<str>) -> procres {
    ret program_output(config, testfile, lib_path,
                       procargs.prog, procargs.args, procenv, input);
}

fn make_compile_args(config: config, props: test_props, extras: [str],
                     xform: fn(config, str) -> str, testfile: str) ->
   procargs {
    let prog = config.rustc_path;
    let mut args = [testfile, "-o", xform(config, testfile),
                    "-L", config.build_base] + extras;
    args += split_maybe_args(config.rustcflags);
    args += split_maybe_args(props.compile_flags);
    ret {prog: prog, args: args};
}

fn make_lib_name(config: config, auxfile: str, testfile: str) -> str {
    // what we return here is not particularly important, as it
    // happens; rustc ignores everything except for the directory.
    let auxname = output_testname(auxfile);
    path::connect(aux_output_dir_name(config, testfile), auxname)
}

fn make_exe_name(config: config, testfile: str) -> str {
    output_base_name(config, testfile) + os::exe_suffix()
}

fn make_run_args(config: config, _props: test_props, testfile: str) ->
   procargs {
    let toolargs = {
            // If we've got another tool to run under (valgrind),
            // then split apart its command
            let runtool =
                alt config.runtool {
                  option::some(s) { option::some(s) }
                  option::none { option::none }
                };
            split_maybe_args(runtool)
        };

    let args = toolargs + [make_exe_name(config, testfile)];
    ret {prog: args[0], args: vec::slice(args, 1u, vec::len(args))};
}

fn split_maybe_args(argstr: option<str>) -> [str] {
    fn rm_whitespace(v: [str]) -> [str] {
        fn flt(&&s: str) -> option<str> {
          if !str::is_whitespace(s) { option::some(s) } else { option::none }
        }
        vec::filter_map(v, flt)
    }

    alt argstr {
      option::some(s) { rm_whitespace(str::split_char(s, ' ')) }
      option::none { [] }
    }
}

fn program_output(config: config, testfile: str, lib_path: str, prog: str,
                  args: [str], env: [(str, str)],
                  input: option<str>) -> procres {
    let cmdline =
        {
            let cmdline = make_cmdline(lib_path, prog, args);
            logv(config, #fmt["executing %s", cmdline]);
            cmdline
        };
    let res = procsrv::run(lib_path, prog, args, env, input);
    dump_output(config, testfile, res.out, res.err);
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

fn aux_output_dir_name(config: config, testfile: str) -> str {
    output_base_name(config, testfile) + ".libaux"
}

fn output_testname(testfile: str) -> str {
    let parts = str::split_char(path::basename(testfile), '.');
    str::connect(vec::slice(parts, 0u, vec::len(parts) - 1u), ".")
}

fn output_base_name(config: config, testfile: str) -> str {
    let base = config.build_base;
    let filename = output_testname(testfile);
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
