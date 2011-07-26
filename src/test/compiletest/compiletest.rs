import std::option;
import std::getopts;
import std::test;
import std::fs;
import std::str;
import std::vec;
import std::ivec;
import std::io;
import std::generic_os::setenv;
import std::generic_os::getenv;
import std::os;
import std::run;
import std::task;

tag mode {
    mode_compile_fail;
    mode_run_fail;
    mode_run_pass;
}

type config = rec(// The library paths required for running the compiler
                  str compile_lib_path,
                  // The library paths required for running compiled programs
                  str run_lib_path,
                  // The rustc executable
                  str rustc_path,
                  // The directory containing the tests to run
                  str src_base,
                  // The directory where programs should be built
                  str build_base,
                  // The name of the stage being built (stage1, etc)
                  str stage_id,
                  // The test mode, compile-fail, run-fail, run-pass
                  mode mode,
                  // Run ignored tests
                  bool run_ignored,
                  // Only run tests that match this filter
                  option::t[str] filter,
                  // A command line to prefix program execution with,
                  // for running under valgrind
                  option::t[str] runtool,
                  // Flags to pass to the compiler
                  option::t[str] rustcflags,
                  // Explain what's going on
                  bool verbose);

fn main(vec[str] args) {

    auto ivec_args = {
        auto ivec_args = ~[];
        for (str arg in args) {
            ivec_args += ~[arg];
        }
        ivec_args
    };

    auto config = parse_config(ivec_args);
    log_config(config);
    run_tests(config);
}

fn parse_config(&str[] args) -> config {
    auto opts = ~[getopts::reqopt("compile-lib-path"),
                  getopts::reqopt("run-lib-path"),
                  getopts::reqopt("rustc-path"),
                  getopts::reqopt("src-base"),
                  getopts::reqopt("build-base"),
                  getopts::reqopt("stage-id"),
                  getopts::reqopt("mode"),
                  getopts::optflag("ignored"),
                  getopts::optopt("runtool"),
                  getopts::optopt("rustcflags"),
                  getopts::optflag("verbose")];

    check ivec::is_not_empty(args);
    auto args_ = ivec::tail(args);
    auto match = alt (getopts::getopts_ivec(args_, opts)) {
        getopts::success(?m) { m }
        getopts::failure(?f) {
            fail getopts::fail_str(f)
        }
    };

    ret rec(compile_lib_path = getopts::opt_str(match, "compile-lib-path"),
            run_lib_path = getopts::opt_str(match, "run-lib-path"),
            rustc_path = getopts::opt_str(match, "rustc-path"),
            src_base = getopts::opt_str(match, "src-base"),
            build_base = getopts::opt_str(match, "build-base"),
            stage_id = getopts::opt_str(match, "stage-id"),
            mode = alt getopts::opt_str(match, "mode") {
                "compile-fail" { mode_compile_fail }
                "run-fail" { mode_run_fail }
                "run-pass" { mode_run_pass }
                _ { fail "invalid mode" }
            },
            run_ignored = getopts::opt_present(match, "ignored"),
            filter = if vec::len(match.free) > 0u {
                option::some(match.free.(0))
            } else {
                option::none
            },
            runtool = getopts::opt_maybe_str(match, "runtool"),
            rustcflags = getopts::opt_maybe_str(match, "rustcflags"),
            verbose = getopts::opt_present(match, "verbose"));
}

fn log_config(&config config) {
    auto c = config;
    logv(c, #fmt("configuration:"));
    logv(c, #fmt("compile_lib_path: %s", config.compile_lib_path));
    logv(c, #fmt("run_lib_path: %s", config.run_lib_path));
    logv(c, #fmt("rustc_path: %s", config.rustc_path));
    logv(c, #fmt("src_base: %s", config.src_base));;
    logv(c, #fmt("build_base: %s", config.build_base));
    logv(c, #fmt("stage_id: %s", config.stage_id));
    logv(c, #fmt("mode: %s", mode_str(config.mode)));
    logv(c, #fmt("run_ignored: %b", config.run_ignored));
    logv(c, #fmt("filter: %s", alt (config.filter) {
      option::some(?f) { f }
      option::none { "(none)" }
    }));
    logv(c, #fmt("runtool: %s", alt (config.runtool) {
      option::some(?s) { s }
      option::none { "(none)" }
    }));
    logv(c, #fmt("rustcflags: %s", alt (config.rustcflags) {
      option::some(?s) { s }
      option::none { "(none)" }
    }));
    logv(c, #fmt("verbose: %b", config.verbose));
    logv(c, #fmt("\n"));
}

fn mode_str(mode mode) -> str {
    alt (mode) {
        mode_compile_fail { "compile-fail" }
        mode_run_fail { "run-fail" }
        mode_run_pass { "run-pass" }
    }
}

type cx = rec(config config,
              procsrv::handle procsrv);

fn run_tests(&config config) {
    auto opts = test_opts(config);
    auto cx = rec(config = config,
                  procsrv = procsrv::mk());
    auto tests = make_tests(cx);
    test::run_tests_console(opts, tests);
    procsrv::close(cx.procsrv);
}

fn test_opts(&config config) -> test::test_opts {
    rec(filter = config.filter,
        run_ignored = config.run_ignored)
}

fn make_tests(&cx cx) -> test::test_desc[] {
    log #fmt("making tests from %s", cx.config.src_base);
    auto tests = ~[];
    for (str file in fs::list_dir(cx.config.src_base)) {
        log #fmt("inspecting file %s", file);
        if (is_test(file)) {
            tests += ~[make_test(cx, file)];
        }
    }
    ret tests;
}

fn is_test(&str testfile) -> bool {
    auto name = fs::basename(testfile);
    (str::ends_with(name, ".rs") || str::ends_with(name, ".rc"))
    && !(str::starts_with(name, ".")
         || str::starts_with(name, "#")
         || str::starts_with(name, "~"))
}

fn make_test(&cx cx, &str testfile) -> test::test_desc {
    rec(name = testfile,
        fn = make_test_fn(cx, testfile),
        ignore = is_test_ignored(cx.config, testfile))
}

fn is_test_ignored(&config config, &str testfile) -> bool {
    auto found = false;
    for each (str ln in iter_header(testfile)) {
        // FIXME: Can't return or break from iterator
        found = found || parse_name_directive(ln, "xfail-" + config.stage_id);
    }
    ret found;
}

iter iter_header(&str testfile) -> str {
    auto rdr = io::file_reader(testfile);
    while !rdr.eof() {
        auto ln = rdr.read_line();
        // Assume that any directives will be found before the
        // first module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if str::starts_with(ln, "fn") || str::starts_with(ln, "mod") {
            break;
        } else {
            put ln;
        }
    }
}

fn make_test_fn(&cx cx, &str testfile) -> test::test_fn {
    // We're doing some ferociously unsafe nonsense here by creating a closure
    // and letting the test runner spawn it into a task. To avoid having
    // different tasks fighting over their refcounts and then the wrong task
    // freeing a box we need to clone everything, and make sure our closure
    // outlives all the tasks.
    fn clonestr(&str s) -> str {
        str::unsafe_from_bytes(str::bytes(s))
    }

    fn cloneoptstr(&option::t[str] s) -> option::t[str] {
        alt s {
          option::some(?s) { option::some(clonestr(s)) }
          option::none { option::none }
        }
    }

    auto configclone = rec(
        compile_lib_path = clonestr(cx.config.compile_lib_path),
        run_lib_path = clonestr(cx.config.run_lib_path),
        rustc_path = clonestr(cx.config.rustc_path),
        src_base = clonestr(cx.config.src_base),
        build_base = clonestr(cx.config.build_base),
        stage_id = clonestr(cx.config.stage_id),
        mode = cx.config.mode,
        run_ignored = cx.config.run_ignored,
        filter = cloneoptstr(cx.config.filter),
        runtool = cloneoptstr(cx.config.runtool),
        rustcflags = cloneoptstr(cx.config.rustcflags),
        verbose = cx.config.verbose);
    auto cxclone = rec(config = configclone,
                       procsrv = procsrv::clone(cx.procsrv));
    auto testfileclone = clonestr(testfile);
    ret bind run_test(cxclone, testfileclone);
}

fn run_test(cx cx, str testfile) {
    log #fmt("running %s", testfile);
    auto props = load_props(testfile);
    alt (cx.config.mode) {
        mode_compile_fail {
            run_cfail_test(cx, props, testfile);
        }
        mode_run_fail {
            run_rfail_test(cx, props, testfile);
        }
        mode_run_pass {
            run_rpass_test(cx, props, testfile);
        }
    }
}

type test_props = rec(str[] error_patterns,
                      option::t[str] compile_flags);

// Load any test directives embedded in the file
fn load_props(&str testfile) -> test_props {
    auto error_patterns = ~[];
    auto compile_flags = option::none;
    for each (str ln in iter_header(testfile)) {
        alt parse_error_pattern(ln) {
          option::some(?ep) { error_patterns += ~[ep]; }
          option::none { }
        }

        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }
    }
    ret rec(error_patterns = error_patterns,
            compile_flags = compile_flags);
}

fn parse_error_pattern(&str line) -> option::t[str] {
    parse_name_value_directive(line, "error-pattern")
}

fn parse_compile_flags(&str line) -> option::t[str] {
    parse_name_value_directive(line, "compile-flags")
}

fn parse_name_directive(&str line, &str directive) -> bool {
    str::find(line, directive) >= 0
}

fn parse_name_value_directive(&str line, &str directive) -> option::t[str] {
    auto keycolon = directive + ":";
    if str::find(line, keycolon) >= 0 {
        auto colon = str::find(line, keycolon) as uint;
        auto value = str::slice(line,
                                colon + str::byte_len(keycolon),
                                str::byte_len(line));
        log #fmt("%s: %s", directive, value);
        option::some(value)
    } else {
        option::none
    }
}

fn run_cfail_test(&cx cx, &test_props props, &str testfile) {
    auto procres = compile_test(cx, props, testfile);

    if (procres.status == 0) {
        fatal_procres("compile-fail test compiled successfully!", procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rfail_test(&cx cx, &test_props props, &str testfile) {
    auto procres = compile_test(cx, props, testfile);

    if (procres.status != 0) {
        fatal_procres("compilation failed!", procres);
    }

    procres = exec_compiled_test(cx, testfile);

    if (procres.status == 0) {
        fatal_procres("run-fail test didn't produce an error!",
                      procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rpass_test(&cx cx, &test_props props, &str testfile) {
    auto procres = compile_test(cx, props, testfile);

    if (procres.status != 0) {
        fatal_procres("compilation failed!", procres);
    }

    procres = exec_compiled_test(cx, testfile);

    if (procres.status != 0) {
        fatal_procres("test run failed!", procres);
    }
}

fn check_error_patterns(&test_props props, &str testfile,
                       &procres procres) {
    if ivec::is_empty(props.error_patterns) {
        fatal("no error pattern specified in " + testfile);
    }

    auto next_err_idx = 0u;
    auto next_err_pat = props.error_patterns.(next_err_idx);
    for (str line in str::split(procres.out, '\n' as u8)) {
        if (str::find(line, next_err_pat) > 0) {
            log #fmt("found error pattern %s", next_err_pat);
            next_err_idx += 1u;
            if next_err_idx == ivec::len(props.error_patterns) {
                log "found all error patterns";
                ret;
            }
            next_err_pat = props.error_patterns.(next_err_idx);
        }
    }

    auto missing_patterns = ivec::slice(props.error_patterns,
                                        next_err_idx,
                                        ivec::len(props.error_patterns));
    if (ivec::len(missing_patterns) == 1u) {
        fatal_procres(#fmt("error pattern '%s' not found!",
                           missing_patterns.(0)),
                      procres);
    } else {
        for (str pattern in missing_patterns) {
            error(#fmt("error pattern '%s' not found!", pattern));
        }
        fatal_procres("multiple error patterns not found", procres);
    }
}

type procargs = rec(str prog, vec[str] args);

type procres = rec(int status, str out, str cmdline);

fn compile_test(&cx cx, &test_props props,
                &str testfile) -> procres {
    compose_and_run(cx,
                    testfile,
                    bind make_compile_args(_, props, _),
                    cx.config.compile_lib_path)
}

fn exec_compiled_test(&cx cx, &str testfile) -> procres {
    compose_and_run(cx,
                    testfile,
                    make_run_args,
                    cx.config.run_lib_path)
}

fn compose_and_run(&cx cx, &str testfile,
                   fn(&config, &str) -> procargs make_args,
                   &str lib_path) -> procres {
    auto procargs = make_args(cx.config, testfile);
    ret program_output(cx, testfile, lib_path,
                       procargs.prog, procargs.args);
}

fn make_compile_args(&config config, &test_props props,
                     &str testfile) -> procargs {
    auto prog = config.rustc_path;
    auto args = [testfile,
                 "-o", make_exe_name(config, testfile)];
    args += split_maybe_args(config.rustcflags);
    args += split_maybe_args(props.compile_flags);
    ret rec(prog = prog,
            args = args);
}

fn make_run_args(&config config, &str testfile) -> procargs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    auto args = split_maybe_args(config.runtool)
        + [make_exe_name(config, testfile)];
    ret rec(prog = args.(0),
            args = vec::slice(args, 1u, vec::len(args)));
}

fn split_maybe_args(&option::t[str] argstr) -> vec[str] {
    alt (argstr) {
        option::some(?s) { str::split(s, ' ' as u8) }
        option::none { [] }
    }
}

fn program_output(&cx cx, &str testfile,
                  &str lib_path, &str prog, &vec[str] args) -> procres {
    auto cmdline = {
        auto cmdline = make_cmdline(lib_path, prog, args);
        logv(cx.config, #fmt("running %s", cmdline));
        cmdline
    };
    auto res = procsrv::run(cx.procsrv, lib_path, prog, args);
    dump_output(cx.config, testfile, res.out);
    ret rec(status = res.status,
            out = res.out,
            cmdline = cmdline);
}

fn make_cmdline(&str libpath, &str prog, &vec[str] args) -> str {
    #fmt("%s %s %s",
         lib_path_cmd_prefix(libpath),
         prog,
         str::connect(args, " "))
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
fn lib_path_cmd_prefix(&str path) -> str {
    #fmt("%s=\"%s\"", lib_path_env_var(), make_new_path(path))
}

fn make_new_path(&str path) -> str {
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    alt getenv(lib_path_env_var()) {
        option::some(?curr) { #fmt("%s:%s", path, curr) }
        option::none { path }
    }
}

#[cfg(target_os = "linux")]
fn lib_path_env_var() -> str { "LD_LIBRARY_PATH" }

#[cfg(target_os = "macos")]
fn lib_path_env_var() -> str { "DYLD_LIBRARY_PATH" }

#[cfg(target_os = "win32")]
fn lib_path_env_var() -> str { "PATH" }

fn make_exe_name(&config config, &str testfile) -> str {
    output_base_name(config, testfile) + os::exec_suffix()
}

fn output_base_name(&config config, &str testfile) -> str {
    auto base = config.build_base;
    auto filename = {
        auto parts = str::split(fs::basename(testfile), '.' as u8);
        parts = vec::slice(parts, 0u, vec::len(parts) - 1u);
        str::connect(parts, ".")
    };
    #fmt("%s%s.%s", base, filename, config.stage_id)
}

#[cfg(target_os = "win32")]
#[cfg(target_os = "linux")]
fn dump_output(&config config, &str testfile, &str out) {
    auto outfile = make_out_name(config, testfile);
    auto writer = io::file_writer(outfile, [io::create, io::truncate]);
    writer.write_str(out);
    maybe_dump_to_stdout(config, out);
}

// FIXME (726): Can't use file_writer on mac
#[cfg(target_os = "macos")]
fn dump_output(&config config, &str testfile, &str out) {
    maybe_dump_to_stdout(config, out);
}

fn maybe_dump_to_stdout(&config config, &str out) {
    if (config.verbose) {
        io::stdout().write_line("------------------------------------------");
        io::stdout().write_line(out);
        io::stdout().write_line("------------------------------------------");
    }
}

fn make_out_name(&config config, &str testfile) -> str {
    output_base_name(config, testfile) + ".out"
}

fn error(&str err) {
    io::stdout().write_line(#fmt("\nerror: %s", err));
}

fn fatal(&str err) -> ! {
    error(err);
    fail;
}

fn fatal_procres(&str err, procres procres) -> ! {
    auto msg = #fmt("\n\
                     error: %s\n\
                     command: %s\n\
                     output:\n\
                     ------------------------------------------\n\
                     %s\n\
                     ------------------------------------------\n\
                     \n",
                    err, procres.cmdline, procres.out);
    io::stdout().write_str(msg);
    fail;
}

fn logv(&config config, &str s) {
    log s;
    if (config.verbose) {
        io::stdout().write_line(s);
    }
}


// So when running tests in parallel there's a potential race on environment
// variables if we let each task spawn its own children - between the time the
// environment is set and the process is spawned another task could spawn its
// child process. Because of that we have to use a complicated scheme with a
// dedicated server for spawning processes.
mod procsrv {

    export handle;
    export mk;
    export clone;
    export run;
    export close;

    type handle = rec(option::t[task] task,
                      chan[request] chan);

    tag request {
        exec(str, str, vec[str], chan[response]);
        stop;
    }

    type response = rec(int pid, int outfd);

    fn mk() -> handle {
        auto res = task::worker(worker);
        ret rec(task = option::some(res.task),
                chan = res.chan);
    }

    fn clone(&handle handle) -> handle {
        // Sharing tasks across tasks appears to be (yet another) recipe for
        // disaster, so our handle clones will not get the task pointer.
        rec(task = option::none,
            chan = task::clone_chan(handle.chan))
    }

    fn close(&handle handle) {
        task::send(handle.chan, stop);
        task::join(option::get(handle.task));
    }

    fn run(&handle handle, &str lib_path,
           &str prog, &vec[str] args) -> rec(int status, str out) {
        auto p = port[response]();
        auto ch = chan(p);
        task::send(handle.chan,
                   exec(lib_path, prog, args, ch));

        auto resp = task::recv(p);
        // Copied from run::program_output
        auto outfile = os::fd_FILE(resp.outfd);
        auto reader = io::new_reader(io::FILE_buf_reader(outfile, false));
        auto buf = "";
        while (!reader.eof()) {
            auto bytes = reader.read_bytes(4096u);
            buf += str::unsafe_from_bytes(bytes);
        }
        os::libc::fclose(outfile);
        ret rec(status = os::waitpid(resp.pid), out = buf);
    }

    fn worker(port[request] p) {
        while (true) {
            alt task::recv(p) {
              exec(?lib_path, ?prog, ?args, ?respchan) {
                // This is copied from run::start_program
                auto pipe_in = os::pipe();
                auto pipe_out = os::pipe();
                auto spawnproc = bind run::spawn_process(
                    prog, args, pipe_in.in, pipe_out.out, 0);
                auto pid = with_lib_path(lib_path, spawnproc);
                if (pid == -1) { fail; }
                os::libc::close(pipe_in.in);
                os::libc::close(pipe_in.out);
                os::libc::close(pipe_out.out);
                task::send(respchan, rec(pid = pid,
                                         outfd = pipe_out.in));
              }
              stop {
                ret;
              }
            }
        }
    }

    fn with_lib_path[T](&str path, fn() -> T f) -> T {
        auto maybe_oldpath = getenv(lib_path_env_var());
        append_lib_path(path);
        auto res = f();
        if option::is_some(maybe_oldpath) {
            export_lib_path(option::get(maybe_oldpath));
        } else {
            // FIXME: This should really be unset but we don't have that yet
            export_lib_path("");
        }
        ret res;
    }

    fn append_lib_path(&str path) {
        export_lib_path(make_new_path(path));
    }

    fn export_lib_path(&str path) {
        setenv(lib_path_env_var(), path);
    }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
