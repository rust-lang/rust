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

tag mode { mode_compile_fail; mode_run_fail; mode_run_pass; }

type config = {
 // The library paths required for running the compiler
 compile_lib_path: str,
 // The library paths required for running compiled programs
 run_lib_path: str,
 // The rustc executable
 rustc_path: str,
 // The directory containing the tests to run
 src_base: str,
 // The directory where programs should be built
 build_base: str,
 // The name of the stage being built (stage1, etc)
 stage_id: str,
 // The test mode, compile-fail, run-fail, run-pass
 mode: mode,
 // Run ignored tests
 run_ignored: bool,
 // Only run tests that match this filter
 filter: option::t[str],
 // A command line to prefix program execution with,
 // for running under valgrind
 runtool: option::t[str],
 // Flags to pass to the compiler
 rustcflags: option::t[str],
 // Explain what's going on
 verbose: bool
};

fn main(args: vec[str]) {

    let ivec_args =
        {
            let ivec_args = ~[];
            for arg: str  in args { ivec_args += ~[arg]; }
            ivec_args
        };

    let config = parse_config(ivec_args);
    log_config(config);
    run_tests(config);
}

fn parse_config(args: &str[]) -> config {
    let opts =
        ~[getopts::reqopt("compile-lib-path"),
          getopts::reqopt("run-lib-path"), getopts::reqopt("rustc-path"),
          getopts::reqopt("src-base"), getopts::reqopt("build-base"),
          getopts::reqopt("stage-id"), getopts::reqopt("mode"),
          getopts::optflag("ignored"), getopts::optopt("runtool"),
          getopts::optopt("rustcflags"), getopts::optflag("verbose")];

    check (ivec::is_not_empty(args));
    let args_ = ivec::tail(args);
    let match =
        alt getopts::getopts_ivec(args_, opts) {
          getopts::success(m) { m }
          getopts::failure(f) { fail getopts::fail_str(f) }
        };

    ret {compile_lib_path: getopts::opt_str(match, "compile-lib-path"),
         run_lib_path: getopts::opt_str(match, "run-lib-path"),
         rustc_path: getopts::opt_str(match, "rustc-path"),
         src_base: getopts::opt_str(match, "src-base"),
         build_base: getopts::opt_str(match, "build-base"),
         stage_id: getopts::opt_str(match, "stage-id"),
         mode: str_mode(getopts::opt_str(match, "mode")),
         run_ignored: getopts::opt_present(match, "ignored"),
         filter:
             if vec::len(match.free) > 0u {
                 option::some(match.free.(0))
             } else { option::none },
         runtool: getopts::opt_maybe_str(match, "runtool"),
         rustcflags: getopts::opt_maybe_str(match, "rustcflags"),
         verbose: getopts::opt_present(match, "verbose")};
}

fn log_config(config: &config) {
    let c = config;
    logv(c, #fmt("configuration:"));
    logv(c, #fmt("compile_lib_path: %s", config.compile_lib_path));
    logv(c, #fmt("run_lib_path: %s", config.run_lib_path));
    logv(c, #fmt("rustc_path: %s", config.rustc_path));
    logv(c, #fmt("src_base: %s", config.src_base));
    logv(c, #fmt("build_base: %s", config.build_base));
    logv(c, #fmt("stage_id: %s", config.stage_id));
    logv(c, #fmt("mode: %s", mode_str(config.mode)));
    logv(c, #fmt("run_ignored: %b", config.run_ignored));
    logv(c, #fmt("filter: %s", opt_str(config.filter)));
    logv(c, #fmt("runtool: %s", opt_str(config.runtool)));
    logv(c, #fmt("rustcflags: %s", opt_str(config.rustcflags)));
    logv(c, #fmt("verbose: %b", config.verbose));
    logv(c, #fmt("\n"));
}

fn opt_str(maybestr: option::t[str]) -> str {
    alt maybestr { option::some(s) { s } option::none. { "(none)" } }
}

fn str_opt(maybestr: str) -> option::t[str] {
    if maybestr != "(none)" { option::some(maybestr) } else { option::none }
}

fn str_mode(s: str) -> mode {
    alt s {
      "compile-fail" { mode_compile_fail }
      "run-fail" { mode_run_fail }
      "run-pass" { mode_run_pass }
      _ { fail "invalid mode" }
    }
}

fn mode_str(mode: mode) -> str {
    alt mode {
      mode_compile_fail. { "compile-fail" }
      mode_run_fail. { "run-fail" }
      mode_run_pass. { "run-pass" }
    }
}

type cx = {config: config, procsrv: procsrv::handle};

fn run_tests(config: &config) {
    let opts = test_opts(config);
    let cx = {config: config, procsrv: procsrv::mk()};
    let tests = make_tests(cx);
    test::run_tests_console_(opts, tests.tests, tests.to_task);
    procsrv::close(cx.procsrv);
}

fn test_opts(config: &config) -> test::test_opts {
    {filter: config.filter, run_ignored: config.run_ignored}
}

type tests_and_conv_fn =
    {tests: test::test_desc[], to_task: fn(&fn() ) -> task };

fn make_tests(cx: &cx) -> tests_and_conv_fn {
    log #fmt("making tests from %s", cx.config.src_base);
    let configport = port[str]();
    let tests = ~[];
    for file: str  in fs::list_dir(cx.config.src_base) {
        log #fmt("inspecting file %s", file);
        if is_test(file) { tests += ~[make_test(cx, file, configport)]; }
    }
    ret {tests: tests, to_task: bind closure_to_task(cx, configport, _)};
}

fn is_test(testfile: &str) -> bool {
    let name = fs::basename(testfile);
    (str::ends_with(name, ".rs") || str::ends_with(name, ".rc")) &&
        !(str::starts_with(name, ".") || str::starts_with(name, "#") ||
              str::starts_with(name, "~"))
}

fn make_test(cx: &cx, testfile: &str, configport: &port[str]) ->
   test::test_desc {
    {name: testfile,
     fn: make_test_closure(testfile, chan(configport)),
     ignore: is_test_ignored(cx.config, testfile)}
}

fn is_test_ignored(config: &config, testfile: &str) -> bool {
    let found = false;
    for each ln: str  in iter_header(testfile) {
        // FIXME: Can't return or break from iterator
        found = found || parse_name_directive(ln, "xfail-" + config.stage_id);
    }
    ret found;
}

iter iter_header(testfile: &str) -> str {
    let rdr = io::file_reader(testfile);
    while !rdr.eof() {
        let ln = rdr.read_line();

        // Assume that any directives will be found before the
        // first module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        if str::starts_with(ln, "fn") || str::starts_with(ln, "mod") {
            break;
        } else { put ln; }
    }
}

/*
So this is kind of crappy:

A test is just defined as a function, as you might expect, but tests have to
run their own tasks. Unfortunately, if your test needs dynamic data then it
needs to be a closure, and transferring closures across tasks without
committing a host of memory management transgressions is just impossible.

To get around this, the standard test runner allows you the opportunity do
your own conversion from a test function to a task. It gives you your function
and you give it back a task.

So that's what we're going to do. Here's where it gets stupid. To get the
the data out of the test function we are going to run the test function,
which will do nothing but send the data for that test to a port we've set
up. Then we'll spawn that data into another task and return the task.
Really convoluted. Need to think up of a better definition for tests.
*/

fn make_test_closure(testfile: &str, configchan: chan[str]) -> test::test_fn {
    bind send_config(testfile, configchan)
}

fn send_config(testfile: str, configchan: chan[str]) {
    task::send(configchan, testfile);
}

/*
FIXME: Good god forgive me.

So actually shuttling structural data across tasks isn't possible at this
time, but we can send strings! Sadly, I need the whole config record, in the
test task so, instead of fixing the mechanism in the compiler I'm going to
break up the config record and pass everything individually to the spawned
function.
*/

fn closure_to_task(cx: cx, configport: port[str], testfn: &fn() ) -> task {
    testfn();
    let testfile = task::recv(configport);
    ret spawn run_test_task(cx.config.compile_lib_path,
                            cx.config.run_lib_path, cx.config.rustc_path,
                            cx.config.src_base, cx.config.build_base,
                            cx.config.stage_id, mode_str(cx.config.mode),
                            cx.config.run_ignored, opt_str(cx.config.filter),
                            opt_str(cx.config.runtool),
                            opt_str(cx.config.rustcflags), cx.config.verbose,
                            procsrv::clone(cx.procsrv).chan, testfile);
}

fn run_test_task(compile_lib_path: str, run_lib_path: str, rustc_path: str,
                 src_base: str, build_base: str, stage_id: str, mode: str,
                 run_ignored: bool, opt_filter: str, opt_runtool: str,
                 opt_rustcflags: str, verbose: bool,
                 procsrv_chan: procsrv::reqchan, testfile: str) {

    let config =
        {compile_lib_path: compile_lib_path,
         run_lib_path: run_lib_path,
         rustc_path: rustc_path,
         src_base: src_base,
         build_base: build_base,
         stage_id: stage_id,
         mode: str_mode(mode),
         run_ignored: run_ignored,
         filter: str_opt(opt_filter),
         runtool: str_opt(opt_runtool),
         rustcflags: str_opt(opt_rustcflags),
         verbose: verbose};

    let procsrv = procsrv::from_chan(procsrv_chan);

    let cx = {config: config, procsrv: procsrv};

    log #fmt("running %s", testfile);
    task::unsupervise();
    let props = load_props(testfile);
    alt cx.config.mode {
      mode_compile_fail. { run_cfail_test(cx, props, testfile); }
      mode_run_fail. { run_rfail_test(cx, props, testfile); }
      mode_run_pass. { run_rpass_test(cx, props, testfile); }
    }
}

type test_props = {error_patterns: str[], compile_flags: option::t[str]};

// Load any test directives embedded in the file
fn load_props(testfile: &str) -> test_props {
    let error_patterns = ~[];
    let compile_flags = option::none;
    for each ln: str  in iter_header(testfile) {
        alt parse_error_pattern(ln) {
          option::some(ep) { error_patterns += ~[ep]; }
          option::none. { }
        }


        if option::is_none(compile_flags) {
            compile_flags = parse_compile_flags(ln);
        }
    }
    ret {error_patterns: error_patterns, compile_flags: compile_flags};
}

fn parse_error_pattern(line: &str) -> option::t[str] {
    parse_name_value_directive(line, "error-pattern")
}

fn parse_compile_flags(line: &str) -> option::t[str] {
    parse_name_value_directive(line, "compile-flags")
}

fn parse_name_directive(line: &str, directive: &str) -> bool {
    str::find(line, directive) >= 0
}

fn parse_name_value_directive(line: &str, directive: &str) -> option::t[str] {
    let keycolon = directive + ":";
    if str::find(line, keycolon) >= 0 {
        let colon = str::find(line, keycolon) as uint;
        let value =
            str::slice(line, colon + str::byte_len(keycolon),
                       str::byte_len(line));
        log #fmt("%s: %s", directive, value);
        option::some(value)
    } else { option::none }
}

fn run_cfail_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status == 0 {
        fatal_procres("compile-fail test compiled successfully!", procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rfail_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, testfile);

    if procres.status == 0 {
        fatal_procres("run-fail test didn't produce an error!", procres);
    }

    check_error_patterns(props, testfile, procres);
}

fn run_rpass_test(cx: &cx, props: &test_props, testfile: &str) {
    let procres = compile_test(cx, props, testfile);

    if procres.status != 0 { fatal_procres("compilation failed!", procres); }

    procres = exec_compiled_test(cx, testfile);


    if procres.status != 0 { fatal_procres("test run failed!", procres); }
}

fn check_error_patterns(props: &test_props, testfile: &str,
                        procres: &procres) {
    if ivec::is_empty(props.error_patterns) {
        fatal("no error pattern specified in " + testfile);
    }

    let next_err_idx = 0u;
    let next_err_pat = props.error_patterns.(next_err_idx);
    for line: str  in str::split(procres.out, '\n' as u8) {
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
        for pattern: str  in missing_patterns {
            error(#fmt("error pattern '%s' not found!", pattern));
        }
        fatal_procres("multiple error patterns not found", procres);
    }
}

type procargs = {prog: str, args: vec[str]};

type procres = {status: int, out: str, cmdline: str};

fn compile_test(cx: &cx, props: &test_props, testfile: &str) -> procres {
    compose_and_run(cx, testfile, bind make_compile_args(_, props, _),
                    cx.config.compile_lib_path)
}

fn exec_compiled_test(cx: &cx, testfile: &str) -> procres {
    compose_and_run(cx, testfile, make_run_args, cx.config.run_lib_path)
}

fn compose_and_run(cx: &cx, testfile: &str,
                   make_args: fn(&config, &str) -> procargs , lib_path: &str)
   -> procres {
    let procargs = make_args(cx.config, testfile);
    ret program_output(cx, testfile, lib_path, procargs.prog, procargs.args);
}

fn make_compile_args(config: &config, props: &test_props, testfile: &str) ->
   procargs {
    let prog = config.rustc_path;
    let args = [testfile, "-o", make_exe_name(config, testfile)];
    args += split_maybe_args(config.rustcflags);
    args += split_maybe_args(props.compile_flags);
    ret {prog: prog, args: args};
}

fn make_run_args(config: &config, testfile: &str) -> procargs {
    // If we've got another tool to run under (valgrind),
    // then split apart its command
    let args =
        split_maybe_args(config.runtool) + [make_exe_name(config, testfile)];
    ret {prog: args.(0), args: vec::slice(args, 1u, vec::len(args))};
}

fn split_maybe_args(argstr: &option::t[str]) -> vec[str] {
    alt argstr {
      option::some(s) { str::split(s, ' ' as u8) }
      option::none. { [] }
    }
}

fn program_output(cx: &cx, testfile: &str, lib_path: &str, prog: &str,
                  args: &vec[str]) -> procres {
    let cmdline =
        {
            let cmdline = make_cmdline(lib_path, prog, args);
            logv(cx.config, #fmt("running %s", cmdline));
            cmdline
        };
    let res = procsrv::run(cx.procsrv, lib_path, prog, args);
    dump_output(cx.config, testfile, res.out);
    ret {status: res.status, out: res.out, cmdline: cmdline};
}

fn make_cmdline(libpath: &str, prog: &str, args: &vec[str]) -> str {
    #fmt("%s %s %s", lib_path_cmd_prefix(libpath), prog,
         str::connect(args, " "))
}

// Build the LD_LIBRARY_PATH variable as it would be seen on the command line
// for diagnostic purposes
fn lib_path_cmd_prefix(path: &str) -> str {
    #fmt("%s=\"%s\"", lib_path_env_var(), make_new_path(path))
}

fn make_new_path(path: &str) -> str {

    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    alt getenv(lib_path_env_var()) {
      option::some(curr) { #fmt("%s:%s", path, curr) }
      option::none. { path }
    }
}

#[cfg(target_os = "linux")]
fn lib_path_env_var() -> str { "LD_LIBRARY_PATH" }

#[cfg(target_os = "macos")]
fn lib_path_env_var() -> str { "DYLD_LIBRARY_PATH" }

#[cfg(target_os = "win32")]
fn lib_path_env_var() -> str { "PATH" }

fn make_exe_name(config: &config, testfile: &str) -> str {
    output_base_name(config, testfile) + os::exec_suffix()
}

fn output_base_name(config: &config, testfile: &str) -> str {
    let base = config.build_base;
    let filename =
        {
            let parts = str::split(fs::basename(testfile), '.' as u8);
            parts = vec::slice(parts, 0u, vec::len(parts) - 1u);
            str::connect(parts, ".")
        };
    #fmt("%s%s.%s", base, filename, config.stage_id)
}

#[cfg(target_os = "win32")]
#[cfg(target_os = "linux")]
fn dump_output(config: &config, testfile: &str, out: &str) {
    let outfile = make_out_name(config, testfile);
    let writer = io::file_writer(outfile, [io::create, io::truncate]);
    writer.write_str(out);
    maybe_dump_to_stdout(config, out);
}

// FIXME (726): Can't use file_writer on mac
#[cfg(target_os = "macos")]
fn dump_output(config: &config, testfile: &str, out: &str) {
    maybe_dump_to_stdout(config, out);
}

fn maybe_dump_to_stdout(config: &config, out: &str) {
    if config.verbose {
        io::stdout().write_line("------------------------------------------");
        io::stdout().write_line(out);
        io::stdout().write_line("------------------------------------------");
    }
}

fn make_out_name(config: &config, testfile: &str) -> str {
    output_base_name(config, testfile) + ".out"
}

fn error(err: &str) { io::stdout().write_line(#fmt("\nerror: %s", err)); }

fn fatal(err: &str) -> ! { error(err); fail; }

fn fatal_procres(err: &str, procres: procres) -> ! {
    let msg =
        #fmt("\n\
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

fn logv(config: &config, s: &str) {
    log s;
    if config.verbose { io::stdout().write_line(s); }
}


// So when running tests in parallel there's a potential race on environment
// variables if we let each task spawn its own children - between the time the
// environment is set and the process is spawned another task could spawn its
// child process. Because of that we have to use a complicated scheme with a
// dedicated server for spawning processes.
mod procsrv {

    export handle;
    export mk;
    export from_chan;
    export clone;
    export run;
    export close;
    export reqchan;

    type reqchan = chan[request];

    type handle = {task: option::t[task], chan: reqchan};

    tag request { exec(str, str, vec[str], chan[response]); stop; }

    type response = {pid: int, outfd: int};

    fn mk() -> handle {
        auto setupport = port();
        auto task = spawn fn(chan[chan[request]] setupchan) {
            auto reqport = port();
            auto reqchan = chan(reqport);
            task::send(setupchan, reqchan);
            worker(reqport);
        } (chan(setupport));
        ret {task: option::some(task),
                chan: task::recv(setupport)
                };
    }

    fn from_chan(ch: &reqchan) -> handle { {task: option::none, chan: ch} }

    fn clone(handle: &handle) -> handle {

        // Sharing tasks across tasks appears to be (yet another) recipe for
        // disaster, so our handle clones will not get the task pointer.
        {task: option::none, chan: task::clone_chan(handle.chan)}
    }

    fn close(handle: &handle) {
        task::send(handle.chan, stop);
        task::join(option::get(handle.task));
    }

    fn run(handle: &handle, lib_path: &str, prog: &str, args: &vec[str]) ->
       {status: int, out: str} {
        let p = port[response]();
        let ch = chan(p);
        task::send(handle.chan, exec(lib_path, prog, args, ch));

        let resp = task::recv(p);
        // Copied from run::program_output
        let outfile = os::fd_FILE(resp.outfd);
        let reader = io::new_reader(io::FILE_buf_reader(outfile, false));
        let buf = "";
        while !reader.eof() {
            let bytes = reader.read_bytes(4096u);
            buf += str::unsafe_from_bytes(bytes);
        }
        os::libc::fclose(outfile);
        ret {status: os::waitpid(resp.pid), out: buf};
    }

    fn worker(p: port[request]) {
        while true {
            alt task::recv(p) {
              exec(lib_path, prog, args, respchan) {
                // This is copied from run::start_program
                let pipe_in = os::pipe();
                let pipe_out = os::pipe();
                let spawnproc =
                    bind run::spawn_process(prog, args, pipe_in.in,
                                            pipe_out.out, 0);
                let pid = with_lib_path(lib_path, spawnproc);
                if pid == -1 { fail; }
                os::libc::close(pipe_in.in);
                os::libc::close(pipe_in.out);
                os::libc::close(pipe_out.out);
                task::send(respchan, {pid: pid, outfd: pipe_out.in});
              }
              stop. { ret; }
            }
        }
    }

    fn with_lib_path[T](path: &str, f: fn() -> T ) -> T {
        let maybe_oldpath = getenv(lib_path_env_var());
        append_lib_path(path);
        let res = f();
        if option::is_some(maybe_oldpath) {
            export_lib_path(option::get(maybe_oldpath));
        } else {
            // FIXME: This should really be unset but we don't have that yet
            export_lib_path("");
        }
        ret res;
    }

    fn append_lib_path(path: &str) { export_lib_path(make_new_path(path)); }

    fn export_lib_path(path: &str) { setenv(lib_path_env_var(), path); }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
