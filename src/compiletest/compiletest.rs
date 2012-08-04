import option;
import std::getopts;
import std::test;
import str;
import vec;
import task;

import core::result;
import result::{ok, err};

import comm::port;
import comm::chan;
import comm::send;
import comm::recv;

import common::config;
import common::mode_run_pass;
import common::mode_run_fail;
import common::mode_compile_fail;
import common::mode_pretty;
import common::mode;
import util::logv;

fn main(args: ~[~str]) {
    let config = parse_config(args);
    log_config(config);
    run_tests(config);
}

fn parse_config(args: ~[~str]) -> config {
    let opts =
        ~[getopts::reqopt(~"compile-lib-path"),
          getopts::reqopt(~"run-lib-path"),
          getopts::reqopt(~"rustc-path"), getopts::reqopt(~"src-base"),
          getopts::reqopt(~"build-base"), getopts::reqopt(~"aux-base"),
          getopts::reqopt(~"stage-id"),
          getopts::reqopt(~"mode"), getopts::optflag(~"ignored"),
          getopts::optopt(~"runtool"), getopts::optopt(~"rustcflags"),
          getopts::optflag(~"verbose"),
          getopts::optopt(~"logfile")];

    assert (vec::is_not_empty(args));
    let args_ = vec::tail(args);
    let matches =
        alt getopts::getopts(args_, opts) {
          ok(m) => m,
          err(f) => fail getopts::fail_str(f)
        };

    return {compile_lib_path: getopts::opt_str(matches, ~"compile-lib-path"),
         run_lib_path: getopts::opt_str(matches, ~"run-lib-path"),
         rustc_path: getopts::opt_str(matches, ~"rustc-path"),
         src_base: getopts::opt_str(matches, ~"src-base"),
         build_base: getopts::opt_str(matches, ~"build-base"),
         aux_base: getopts::opt_str(matches, ~"aux-base"),
         stage_id: getopts::opt_str(matches, ~"stage-id"),
         mode: str_mode(getopts::opt_str(matches, ~"mode")),
         run_ignored: getopts::opt_present(matches, ~"ignored"),
         filter:
             if vec::len(matches.free) > 0u {
                 option::some(matches.free[0])
             } else { option::none },
         logfile: getopts::opt_maybe_str(matches, ~"logfile"),
         runtool: getopts::opt_maybe_str(matches, ~"runtool"),
         rustcflags: getopts::opt_maybe_str(matches, ~"rustcflags"),
         verbose: getopts::opt_present(matches, ~"verbose")};
}

fn log_config(config: config) {
    let c = config;
    logv(c, fmt!{"configuration:"});
    logv(c, fmt!{"compile_lib_path: %s", config.compile_lib_path});
    logv(c, fmt!{"run_lib_path: %s", config.run_lib_path});
    logv(c, fmt!{"rustc_path: %s", config.rustc_path});
    logv(c, fmt!{"src_base: %s", config.src_base});
    logv(c, fmt!{"build_base: %s", config.build_base});
    logv(c, fmt!{"stage_id: %s", config.stage_id});
    logv(c, fmt!{"mode: %s", mode_str(config.mode)});
    logv(c, fmt!{"run_ignored: %b", config.run_ignored});
    logv(c, fmt!{"filter: %s", opt_str(config.filter)});
    logv(c, fmt!{"runtool: %s", opt_str(config.runtool)});
    logv(c, fmt!{"rustcflags: %s", opt_str(config.rustcflags)});
    logv(c, fmt!{"verbose: %b", config.verbose});
    logv(c, fmt!{"\n"});
}

fn opt_str(maybestr: option<~str>) -> ~str {
    alt maybestr { option::some(s) => s, option::none => ~"(none)" }
}

fn str_opt(maybestr: ~str) -> option<~str> {
    if maybestr != ~"(none)" { option::some(maybestr) } else { option::none }
}

fn str_mode(s: ~str) -> mode {
    alt s {
      ~"compile-fail" => mode_compile_fail,
      ~"run-fail" => mode_run_fail,
      ~"run-pass" => mode_run_pass,
      ~"pretty" => mode_pretty,
      _ => fail ~"invalid mode"
    }
}

fn mode_str(mode: mode) -> ~str {
    alt mode {
      mode_compile_fail => ~"compile-fail",
      mode_run_fail => ~"run-fail",
      mode_run_pass => ~"run-pass",
      mode_pretty => ~"pretty"
    }
}

fn run_tests(config: config) {
    let opts = test_opts(config);
    let tests = make_tests(config);
    let res = test::run_tests_console(opts, tests);
    if !res { fail ~"Some tests failed"; }
}

fn test_opts(config: config) -> test::test_opts {
    {filter:
         alt config.filter {
           option::some(s) => option::some(s),
           option::none => option::none
         },
     run_ignored: config.run_ignored,
     logfile:
         alt config.logfile {
           option::some(s) => option::some(s),
           option::none => option::none
         }
    }
}

fn make_tests(config: config) -> ~[test::test_desc] {
    debug!{"making tests from %s", config.src_base};
    let mut tests = ~[];
    for os::list_dir_path(config.src_base).each |file| {
        let file = file;
        debug!{"inspecting file %s", file};
        if is_test(config, file) {
            vec::push(tests, make_test(config, file))
        }
    }
    return tests;
}

fn is_test(config: config, testfile: ~str) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        alt config.mode {
          mode_pretty => ~[~".rs"],
          _ => ~[~".rc", ~".rs"]
        };
    let invalid_prefixes = ~[~".", ~"#", ~"~"];
    let name = path::basename(testfile);

    let mut valid = false;

    for valid_extensions.each |ext| {
        if str::ends_with(name, ext) { valid = true; }
    }

    for invalid_prefixes.each |pre| {
        if str::starts_with(name, pre) { valid = false; }
    }

    return valid;
}

fn make_test(config: config, testfile: ~str) ->
   test::test_desc {
    {
        name: make_test_name(config, testfile),
        fn: make_test_closure(config, testfile),
        ignore: header::is_test_ignored(config, testfile),
        should_fail: false
    }
}

fn make_test_name(config: config, testfile: ~str) -> ~str {
    fmt!{"[%s] %s", mode_str(config.mode), testfile}
}

fn make_test_closure(config: config, testfile: ~str) -> test::test_fn {
    fn~() { runtest::run(config, copy testfile) }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
