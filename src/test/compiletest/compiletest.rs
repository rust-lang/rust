import std::option;
import std::getopts;
import std::test;
import std::fs;
import std::str;
import std::vec;
import std::task;

import std::comm;
import std::comm::port;
import std::comm::chan;
import std::comm::send;
import std::comm::recv;

import common::cx;
import common::config;
import common::mode_run_pass;
import common::mode_run_fail;
import common::mode_compile_fail;
import common::mode_pretty;
import common::mode;
import util::logv;

fn main(args: [str]) {
    let config = parse_config(args);
    log_config(config);
    run_tests(config);
}

fn parse_config(args: [str]) -> config {
    let opts =
        [getopts::reqopt("compile-lib-path"), getopts::reqopt("run-lib-path"),
         getopts::reqopt("rustc-path"), getopts::reqopt("src-base"),
         getopts::reqopt("build-base"), getopts::reqopt("stage-id"),
         getopts::reqopt("mode"), getopts::optflag("ignored"),
         getopts::optopt("runtool"), getopts::optopt("rustcflags"),
         getopts::optflag("verbose")];

    check (vec::is_not_empty(args));
    let args_ = vec::tail(args);
    let match =
        alt getopts::getopts(args_, opts) {
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
                 option::some(match.free[0])
             } else { option::none },
         runtool: getopts::opt_maybe_str(match, "runtool"),
         rustcflags: getopts::opt_maybe_str(match, "rustcflags"),
         verbose: getopts::opt_present(match, "verbose")};
}

fn log_config(config: config) {
    let c = config;
    logv(c, #fmt["configuration:"]);
    logv(c, #fmt["compile_lib_path: %s", config.compile_lib_path]);
    logv(c, #fmt["run_lib_path: %s", config.run_lib_path]);
    logv(c, #fmt["rustc_path: %s", config.rustc_path]);
    logv(c, #fmt["src_base: %s", config.src_base]);
    logv(c, #fmt["build_base: %s", config.build_base]);
    logv(c, #fmt["stage_id: %s", config.stage_id]);
    logv(c, #fmt["mode: %s", mode_str(config.mode)]);
    logv(c, #fmt["run_ignored: %b", config.run_ignored]);
    logv(c, #fmt["filter: %s", opt_str(config.filter)]);
    logv(c, #fmt["runtool: %s", opt_str(config.runtool)]);
    logv(c, #fmt["rustcflags: %s", opt_str(config.rustcflags)]);
    logv(c, #fmt["verbose: %b", config.verbose]);
    logv(c, #fmt["\n"]);
}

fn opt_str(maybestr: option::t<str>) -> str {
    alt maybestr { option::some(s) { s } option::none. { "(none)" } }
}

fn str_opt(maybestr: str) -> option::t<str> {
    if maybestr != "(none)" { option::some(maybestr) } else { option::none }
}

fn str_mode(s: str) -> mode {
    alt s {
      "compile-fail" { mode_compile_fail }
      "run-fail" { mode_run_fail }
      "run-pass" { mode_run_pass }
      "pretty" { mode_pretty }
      _ { fail "invalid mode" }
    }
}

fn mode_str(mode: mode) -> str {
    alt mode {
      mode_compile_fail. { "compile-fail" }
      mode_run_fail. { "run-fail" }
      mode_run_pass. { "run-pass" }
      mode_pretty. { "pretty" }
    }
}

fn run_tests(config: config) {
    let opts = test_opts(config);
    let cx = {config: config, procsrv: procsrv::mk()};
    let tests = make_tests(cx);
    let res = test::run_tests_console_(opts, tests.tests, tests.to_task);
    procsrv::close(cx.procsrv);
    if !res { fail "Some tests failed"; }
}

fn test_opts(config: config) -> test::test_opts {
    {filter:
         alt config.filter {
           option::some(s) { option::some(s) }
           option::none. { option::none }
         },
     run_ignored: config.run_ignored}
}

type tests_and_conv_fn =
    {tests: [test::test_desc], to_task: fn(fn()) -> test::joinable};

fn make_tests(cx: cx) -> tests_and_conv_fn {
    log #fmt["making tests from %s", cx.config.src_base];
    let configport = port::<[u8]>();
    let tests = [];
    for file: str in fs::list_dir(cx.config.src_base) {
        let file = file;
        log #fmt["inspecting file %s", file];
        if is_test(cx.config, file) {
            tests += [make_test(cx, file, configport)];
        }
    }
    ret {tests: tests, to_task: bind closure_to_task(cx, configport, _)};
}

fn is_test(config: config, testfile: str) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        alt config.mode { mode_pretty. { [".rs"] } _ { [".rc", ".rs"] } };
    let invalid_prefixes = [".", "#", "~"];
    let name = fs::basename(testfile);

    let valid = false;

    for ext in valid_extensions {
        if str::ends_with(name, ext) { valid = true }
    }

    for pre in invalid_prefixes {
        if str::starts_with(name, pre) { valid = false }
    }

    ret valid;
}

fn make_test(cx: cx, testfile: str, configport: port<[u8]>) ->
   test::test_desc {
    {name: make_test_name(cx.config, testfile),
     fn: make_test_closure(testfile, chan(configport)),
     ignore: header::is_test_ignored(cx.config, testfile)}
}

fn make_test_name(config: config, testfile: str) -> str {
    #fmt["[%s] %s", mode_str(config.mode), testfile]
}

/*
So this is kind of crappy:

A test is just defined as a function, as you might expect, but tests have to
run in their own tasks. Unfortunately, if your test needs dynamic data then it
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

fn make_test_closure(testfile: str, configchan: chan<[u8]>) -> test::test_fn {
    bind send_config(testfile, configchan)
}

fn send_config(testfile: str, configchan: chan<[u8]>) {
    send(configchan, str::bytes(testfile));
}

/*
FIXME: Good god forgive me.

So actually shuttling structural data across tasks isn't possible at this
time, but we can send strings! Sadly, I need the whole config record, in the
test task so, instead of fixing the mechanism in the compiler I'm going to
break up the config record and pass everything individually to the spawned
function.
*/

fn closure_to_task(cx: cx, configport: port<[u8]>, testfn: fn()) ->
   test::joinable {
    testfn();
    let testfile = recv(configport);

    let compile_lib_path = cx.config.compile_lib_path;
    let run_lib_path = cx.config.run_lib_path;
    let rustc_path = cx.config.rustc_path;
    let src_base = cx.config.src_base;
    let build_base = cx.config.build_base;
    let stage_id = cx.config.stage_id;
    let mode = mode_str(cx.config.mode);
    let run_ignored = cx.config.run_ignored;
    let filter = opt_str(cx.config.filter);
    let runtool = opt_str(cx.config.runtool);
    let rustcflags = opt_str(cx.config.rustcflags);
    let verbose = cx.config.verbose;
    let chan = cx.procsrv.chan;

    let testthunk =
        bind run_test_task(compile_lib_path, run_lib_path, rustc_path,
                           src_base, build_base, stage_id, mode, run_ignored,
                           filter, runtool, rustcflags, verbose, chan,
                           testfile);
    ret task::spawn_joinable(testthunk);
}

fn run_test_task(-compile_lib_path: str, -run_lib_path: str, -rustc_path: str,
                 -src_base: str, -build_base: str, -stage_id: str, -mode: str,
                 -run_ignored: bool, -opt_filter: str, -opt_runtool: str,
                 -opt_rustcflags: str, -verbose: bool,
                 -procsrv_chan: procsrv::reqchan, -testfile: [u8]) {

    test::configure_test_task();

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

    runtest::run(cx, testfile);
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
