import option;
import std::getopts;
import std::test;
import str;
import vec;
import task;

import core::result;
import result::{Ok, Err};

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
          getopts::optopt(~"logfile"),
          getopts::optflag(~"jit")];

    assert (vec::is_not_empty(args));
    let args_ = vec::tail(args);
    let matches =
        match getopts::getopts(args_, opts) {
          Ok(m) => m,
          Err(f) => fail getopts::fail_str(f)
        };

    fn opt_path(m: getopts::Matches, nm: ~str) -> Path {
        Path(getopts::opt_str(m, nm))
    }

    return {compile_lib_path: getopts::opt_str(matches, ~"compile-lib-path"),
         run_lib_path: getopts::opt_str(matches, ~"run-lib-path"),
         rustc_path: opt_path(matches, ~"rustc-path"),
         src_base: opt_path(matches, ~"src-base"),
         build_base: opt_path(matches, ~"build-base"),
         aux_base: opt_path(matches, ~"aux-base"),
         stage_id: getopts::opt_str(matches, ~"stage-id"),
         mode: str_mode(getopts::opt_str(matches, ~"mode")),
         run_ignored: getopts::opt_present(matches, ~"ignored"),
         filter:
             if vec::len(matches.free) > 0u {
                 option::Some(matches.free[0])
             } else { option::None },
         logfile: option::map(getopts::opt_maybe_str(matches,
                                                     ~"logfile"),
                              |s| Path(s)),
         runtool: getopts::opt_maybe_str(matches, ~"runtool"),
         rustcflags: getopts::opt_maybe_str(matches, ~"rustcflags"),
         jit: getopts::opt_present(matches, ~"jit"),
         verbose: getopts::opt_present(matches, ~"verbose")};
}

fn log_config(config: config) {
    let c = config;
    logv(c, fmt!("configuration:"));
    logv(c, fmt!("compile_lib_path: %s", config.compile_lib_path));
    logv(c, fmt!("run_lib_path: %s", config.run_lib_path));
    logv(c, fmt!("rustc_path: %s", config.rustc_path.to_str()));
    logv(c, fmt!("src_base: %s", config.src_base.to_str()));
    logv(c, fmt!("build_base: %s", config.build_base.to_str()));
    logv(c, fmt!("stage_id: %s", config.stage_id));
    logv(c, fmt!("mode: %s", mode_str(config.mode)));
    logv(c, fmt!("run_ignored: %b", config.run_ignored));
    logv(c, fmt!("filter: %s", opt_str(config.filter)));
    logv(c, fmt!("runtool: %s", opt_str(config.runtool)));
    logv(c, fmt!("rustcflags: %s", opt_str(config.rustcflags)));
    logv(c, fmt!("jit: %b", config.jit));
    logv(c, fmt!("verbose: %b", config.verbose));
    logv(c, fmt!("\n"));
}

fn opt_str(maybestr: Option<~str>) -> ~str {
    match maybestr { option::Some(s) => s, option::None => ~"(none)" }
}

fn str_opt(maybestr: ~str) -> Option<~str> {
    if maybestr != ~"(none)" { option::Some(maybestr) } else { option::None }
}

fn str_mode(s: ~str) -> mode {
    match s {
      ~"compile-fail" => mode_compile_fail,
      ~"run-fail" => mode_run_fail,
      ~"run-pass" => mode_run_pass,
      ~"pretty" => mode_pretty,
      _ => fail ~"invalid mode"
    }
}

fn mode_str(mode: mode) -> ~str {
    match mode {
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
         match config.filter {
           option::Some(s) => option::Some(s),
           option::None => option::None
         },
     run_ignored: config.run_ignored,
     logfile:
         match config.logfile {
           option::Some(s) => option::Some(s.to_str()),
           option::None => option::None
         }
    }
}

fn make_tests(config: config) -> ~[test::test_desc] {
    debug!("making tests from %s",
           config.src_base.to_str());
    let mut tests = ~[];
    for os::list_dir_path(&config.src_base).each |file| {
        let file = copy file;
        debug!("inspecting file %s", file.to_str());
        if is_test(config, file) {
            vec::push(tests, make_test(config, file))
        }
    }
    return tests;
}

fn is_test(config: config, testfile: &Path) -> bool {
    // Pretty-printer does not work with .rc files yet
    let valid_extensions =
        match config.mode {
          mode_pretty => ~[~".rs"],
          _ => ~[~".rc", ~".rs"]
        };
    let invalid_prefixes = ~[~".", ~"#", ~"~"];
    let name = option::get(testfile.filename());

    let mut valid = false;

    for valid_extensions.each |ext| {
        if str::ends_with(name, ext) { valid = true; }
    }

    for invalid_prefixes.each |pre| {
        if str::starts_with(name, pre) { valid = false; }
    }

    return valid;
}

fn make_test(config: config, testfile: &Path) ->
   test::test_desc {
    {
        name: make_test_name(config, testfile),
        fn: make_test_closure(config, testfile),
        ignore: header::is_test_ignored(config, testfile),
        should_fail: false
    }
}

fn make_test_name(config: config, testfile: &Path) -> ~str {
    fmt!("[%s] %s", mode_str(config.mode), testfile.to_str())
}

fn make_test_closure(config: config, testfile: &Path) -> test::test_fn {
    let testfile = testfile.to_str();
    fn~() { runtest::run(config, testfile) }
}

// Local Variables:
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
