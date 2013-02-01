// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

use getopts;
use sort;
use term;

use core::cmp::Eq;
use core::either::Either;
use core::either;
use core::io::WriterUtil;
use core::io;
use core::libc::size_t;
use core::pipes::{stream, Chan, Port, SharedChan};
use core::option;
use core::prelude::*;
use core::result;
use core::str;
use core::task::TaskBuilder;
use core::task;
use core::vec;

#[abi = "cdecl"]
extern mod rustrt {
    pub unsafe fn rust_sched_threads() -> size_t;
}

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.
pub type TestName = ~str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
pub type TestFn = ~fn();

// The definition of a single test. A test runner will run a list of
// these.
pub struct TestDesc {
    name: TestName,
    ignore: bool,
    should_fail: bool
}

pub struct TestDescAndFn {
    desc: TestDesc,
    testfn: TestFn,
}

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
pub fn test_main(args: &[~str], tests: ~[TestDescAndFn]) {
    let opts =
        match parse_opts(args) {
          either::Left(move o) => o,
          either::Right(move m) => die!(m)
        };
    if !run_tests_console(&opts, tests) { die!(~"Some tests failed"); }
}

pub struct TestOpts {
    filter: Option<~str>,
    run_ignored: bool,
    logfile: Option<~str>,
}

type OptRes = Either<TestOpts, ~str>;

// Parses command line arguments into test options
pub fn parse_opts(args: &[~str]) -> OptRes {
    let args_ = vec::tail(args);
    let opts = ~[getopts::optflag(~"ignored"), getopts::optopt(~"logfile")];
    let matches =
        match getopts::getopts(args_, opts) {
          Ok(move m) => m,
          Err(move f) => return either::Right(getopts::fail_str(f))
        };

    let filter =
        if vec::len(matches.free) > 0 {
            option::Some(matches.free[0])
        } else { option::None };

    let run_ignored = getopts::opt_present(&matches, ~"ignored");
    let logfile = getopts::opt_maybe_str(&matches, ~"logfile");

    let test_opts = TestOpts {
        filter: filter,
        run_ignored: run_ignored,
        logfile: logfile,
    };

    either::Left(test_opts)
}

#[deriving_eq]
pub enum TestResult { TrOk, TrFailed, TrIgnored, }

struct ConsoleTestState {
    out: io::Writer,
    log_out: Option<io::Writer>,
    use_color: bool,
    mut total: uint,
    mut passed: uint,
    mut failed: uint,
    mut ignored: uint,
    mut failures: ~[TestDesc]
}

// A simple console test runner
pub fn run_tests_console(opts: &TestOpts,
                         tests: ~[TestDescAndFn]) -> bool {

    fn callback(event: &TestEvent, st: @ConsoleTestState) {
        debug!("callback(event=%?)", event);
        match *event {
          TeFiltered(ref filtered_tests) => {
            st.total = filtered_tests.len();
            let noun = if st.total != 1 { ~"tests" } else { ~"test" };
            st.out.write_line(fmt!("\nrunning %u %s", st.total, noun));
          }
          TeWait(ref test) => st.out.write_str(
              fmt!("test %s ... ", test.name)),
          TeResult(copy test, result) => {
            match st.log_out {
                Some(f) => write_log(f, result, &test),
                None => ()
            }
            match result {
              TrOk => {
                st.passed += 1;
                write_ok(st.out, st.use_color);
                st.out.write_line(~"");
              }
              TrFailed => {
                st.failed += 1;
                write_failed(st.out, st.use_color);
                st.out.write_line(~"");
                st.failures.push(move test);
              }
              TrIgnored => {
                st.ignored += 1;
                write_ignored(st.out, st.use_color);
                st.out.write_line(~"");
              }
            }
          }
        }
    }

    let log_out = match opts.logfile {
        Some(ref path) => match io::file_writer(&Path(*path),
                                            ~[io::Create, io::Truncate]) {
          result::Ok(w) => Some(w),
          result::Err(ref s) => {
              die!(fmt!("can't open output file: %s", *s))
          }
        },
        None => None
    };

    let st =
        @ConsoleTestState{out: io::stdout(),
          log_out: log_out,
          use_color: use_color(),
          mut total: 0,
          mut passed: 0,
          mut failed: 0,
          mut ignored: 0,
          mut failures: ~[]};

    run_tests(opts, tests, |x| callback(&x, st));

    assert (st.passed + st.failed + st.ignored == st.total);
    let success = st.failed == 0;

    if !success {
        print_failures(st);
    }

    st.out.write_str(fmt!("\nresult: "));
    if success {
        // There's no parallelism at this point so it's safe to use color
        write_ok(st.out, true);
    } else { write_failed(st.out, true); }
    st.out.write_str(fmt!(". %u passed; %u failed; %u ignored\n\n", st.passed,
                          st.failed, st.ignored));

    return success;

    fn write_log(out: io::Writer, result: TestResult, test: &TestDesc) {
        out.write_line(fmt!("%s %s",
                    match result {
                        TrOk => ~"ok",
                        TrFailed => ~"failed",
                        TrIgnored => ~"ignored"
                    }, test.name));
    }

    fn write_ok(out: io::Writer, use_color: bool) {
        write_pretty(out, ~"ok", term::color_green, use_color);
    }

    fn write_failed(out: io::Writer, use_color: bool) {
        write_pretty(out, ~"FAILED", term::color_red, use_color);
    }

    fn write_ignored(out: io::Writer, use_color: bool) {
        write_pretty(out, ~"ignored", term::color_yellow, use_color);
    }

    fn write_pretty(out: io::Writer, word: &str, color: u8, use_color: bool) {
        if use_color && term::color_supported() {
            term::fg(out, color);
        }
        out.write_str(word);
        if use_color && term::color_supported() {
            term::reset(out);
        }
    }
}

fn print_failures(st: @ConsoleTestState) {
    st.out.write_line(~"\nfailures:");
    let failures = copy st.failures;
    let failures = vec::map(failures, |test| test.name);
    let failures = do sort::merge_sort(failures) |x, y| { str::le(*x, *y) };
    for vec::each(failures) |name| {
        st.out.write_line(fmt!("    %s", *name));
    }
}

#[test]
fn should_sort_failures_before_printing_them() {
    fn dummy() {}

    let s = do io::with_str_writer |wr| {
        let test_a = TestDesc {
            name: ~"a",
            ignore: false,
            should_fail: false
        };

        let test_b = TestDesc {
            name: ~"b",
            ignore: false,
            should_fail: false
        };

        let st =
            @ConsoleTestState{out: wr,
              log_out: option::None,
              use_color: false,
              mut total: 0,
              mut passed: 0,
              mut failed: 0,
              mut ignored: 0,
              mut failures: ~[move test_b, move test_a]};

        print_failures(st);
    };

    let apos = str::find_str(s, ~"a").get();
    let bpos = str::find_str(s, ~"b").get();
    assert apos < bpos;
}

fn use_color() -> bool { return get_concurrency() == 1; }

enum TestEvent {
    TeFiltered(~[TestDesc]),
    TeWait(TestDesc),
    TeResult(TestDesc, TestResult),
}

type MonitorMsg = (TestDesc, TestResult);

fn run_tests(opts: &TestOpts,
             tests: ~[TestDescAndFn],
             callback: fn@(e: TestEvent)) {
    let mut filtered_tests = filter_tests(opts, tests);

    let filtered_descs = filtered_tests.map(|t| t.desc);
    callback(TeFiltered(filtered_descs));

    // It's tempting to just spawn all the tests at once, but since we have
    // many tests that run in other processes we would be making a big mess.
    let concurrency = get_concurrency();
    debug!("using %u test tasks", concurrency);

    let mut remaining = filtered_tests;
    vec::reverse(remaining);
    let mut pending = 0;

    let (p, ch) = stream();
    let ch = SharedChan(ch);

    while pending > 0 || !remaining.is_empty() {
        while pending < concurrency && !remaining.is_empty() {
            let test = remaining.pop();
            if concurrency == 1 {
                // We are doing one test at a time so we can print the name
                // of the test before we run it. Useful for debugging tests
                // that hang forever.
                callback(TeWait(test.desc));
            }
            run_test(test, ch.clone());
            pending += 1;
        }

        let (desc, result) = p.recv();
        if concurrency != 1 {
            callback(TeWait(desc));
        }
        callback(TeResult(desc, result));
        pending -= 1;
    }
}

// Windows tends to dislike being overloaded with threads.
#[cfg(windows)]
const sched_overcommit : uint = 1;

#[cfg(unix)]
const sched_overcommit : uint = 4u;

fn get_concurrency() -> uint {
    unsafe {
        let threads = rustrt::rust_sched_threads() as uint;
        if threads == 1 { 1 }
        else { threads * sched_overcommit }
    }
}

#[allow(non_implicitly_copyable_typarams)]
pub fn filter_tests(
    opts: &TestOpts,
    tests: ~[TestDescAndFn]) -> ~[TestDescAndFn]
{
    let mut filtered = tests;

    // Remove tests that don't match the test filter
    filtered = if opts.filter.is_none() {
        move filtered
    } else {
        let filter_str =
            match opts.filter {
          option::Some(copy f) => f,
          option::None => ~""
        };

        fn filter_fn(test: TestDescAndFn, filter_str: &str) ->
            Option<TestDescAndFn> {
            if str::contains(test.desc.name, filter_str) {
                return option::Some(test);
            } else { return option::None; }
        }

        vec::filter_map(filtered, |x| filter_fn(x, filter_str))
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        move filtered
    } else {
        fn filter(test: TestDescAndFn) -> Option<TestDescAndFn> {
            if test.desc.ignore {
                let TestDescAndFn {desc, testfn} = test;
                Some(TestDescAndFn {
                    desc: TestDesc {ignore: false, ..desc},
                    testfn: testfn
                })
            } else {
                None
            }
        };

        vec::filter_map(filtered, |x| filter(x))
    };

    // Sort the tests alphabetically
    pure fn lteq(t1: &TestDescAndFn, t2: &TestDescAndFn) -> bool {
        str::le(t1.desc.name, t2.desc.name)
    }
    sort::quick_sort(filtered, lteq);

    move filtered
}

struct TestFuture {
    test: TestDesc,
    wait: fn@() -> TestResult,
}

pub fn run_test(test: TestDescAndFn, monitor_ch: SharedChan<MonitorMsg>) {
    let TestDescAndFn {desc, testfn} = test;

    if desc.ignore {
        monitor_ch.send((desc, TrIgnored));
        return;
    }

    let testfn_cell = ::cell::Cell(testfn);
    do task::spawn {
        let mut result_future = None; // task::future_result(builder);
        task::task().unlinked().future_result(|+r| {
            result_future = Some(move r);
        }).spawn(testfn_cell.take());
        let task_result = option::unwrap(move result_future).recv();
        let test_result = calc_result(&desc, task_result == task::Success);
        monitor_ch.send((desc, test_result));
    };
}

fn calc_result(desc: &TestDesc, task_succeeded: bool) -> TestResult {
    if task_succeeded {
        if desc.should_fail { TrFailed }
        else { TrOk }
    } else {
        if desc.should_fail { TrOk }
        else { TrFailed }
    }
}

#[cfg(test)]
mod tests {
    use test::{TrFailed, TrIgnored, TrOk, filter_tests, parse_opts,
               TestDesc, TestDescAndFn};
    use test::{TestOpts, run_test};

    use core::either;
    use core::pipes::{stream, SharedChan};
    use core::option;
    use core::vec;

    #[test]
    pub fn do_not_run_ignored_tests() {
        fn f() { die!(); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: ~"whatever",
                ignore: true,
                should_fail: false
            },
            testfn: f,
        };
        let (p, ch) = stream();
        let ch = SharedChan(ch);
        run_test(desc, ch);
        let (_, res) = p.recv();
        assert res != TrOk;
    }

    #[test]
    pub fn ignored_tests_result_in_ignored() {
        fn f() { }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: ~"whatever",
                ignore: true,
                should_fail: false
            },
            testfn: f,
        };
        let (p, ch) = stream();
        let ch = SharedChan(ch);
        run_test(desc, ch);
        let (_, res) = p.recv();
        assert res == TrIgnored;
    }

    #[test]
    #[ignore(cfg(windows))]
    pub fn test_should_fail() {
        fn f() { die!(); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: ~"whatever",
                ignore: false,
                should_fail: true
            },
            testfn: f,
        };
        let (p, ch) = stream();
        let ch = SharedChan(ch);
        run_test(desc, ch);
        let (_, res) = p.recv();
        assert res == TrOk;
    }

    #[test]
    pub fn test_should_fail_but_succeeds() {
        fn f() { }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: ~"whatever",
                ignore: false,
                should_fail: true
            },
            testfn: f,
        };
        let (p, ch) = stream();
        let ch = SharedChan(ch);
        run_test(desc, ch);
        let (_, res) = p.recv();
        assert res == TrFailed;
    }

    #[test]
    pub fn first_free_arg_should_be_a_filter() {
        let args = ~[~"progname", ~"filter"];
        let opts = match parse_opts(args) {
          either::Left(copy o) => o,
          _ => die!(~"Malformed arg in first_free_arg_should_be_a_filter")
        };
        assert ~"filter" == opts.filter.get();
    }

    #[test]
    pub fn parse_ignored_flag() {
        let args = ~[~"progname", ~"filter", ~"--ignored"];
        let opts = match parse_opts(args) {
          either::Left(copy o) => o,
          _ => die!(~"Malformed arg in parse_ignored_flag")
        };
        assert (opts.run_ignored);
    }

    #[test]
    pub fn filter_for_ignored_option() {
        fn dummy() {}

        // When we run ignored tests the test filter should filter out all the
        // unignored tests and flip the ignore flag on the rest to false

        let opts = TestOpts {
            filter: option::None,
            run_ignored: true,
            logfile: option::None,
        };

        let tests = ~[
            TestDescAndFn {
                desc: TestDesc {
                    name: ~"1",
                    ignore: true,
                    should_fail: false,
                },
                testfn: dummy,
            },
            TestDescAndFn {
                desc: TestDesc {
                    name: ~"2",
                    ignore: false,
                    should_fail: false
                },
                testfn: dummy,
            },
        ];
        let filtered = filter_tests(&opts, tests);

        assert (vec::len(filtered) == 1);
        assert (filtered[0].desc.name == ~"1");
        assert (filtered[0].desc.ignore == false);
    }

    #[test]
    pub fn sort_tests() {
        let opts = TestOpts {
            filter: option::None,
            run_ignored: false,
            logfile: option::None,
        };

        let names =
            ~[~"sha1::test", ~"int::test_to_str", ~"int::test_pow",
             ~"test::do_not_run_ignored_tests",
             ~"test::ignored_tests_result_in_ignored",
             ~"test::first_free_arg_should_be_a_filter",
             ~"test::parse_ignored_flag", ~"test::filter_for_ignored_option",
             ~"test::sort_tests"];
        let tests =
        {
            fn testfn() { }
            let mut tests = ~[];
            for vec::each(names) |name| {
                let test = TestDescAndFn {
                    desc: TestDesc {
                        name: *name, ignore: false,
                        should_fail: false
                    },
                    testfn: testfn,
                };
                tests.push(move test);
            }
            move tests
        };
        let filtered = filter_tests(&opts, tests);

        let expected =
            ~[~"int::test_pow", ~"int::test_to_str", ~"sha1::test",
              ~"test::do_not_run_ignored_tests",
              ~"test::filter_for_ignored_option",
              ~"test::first_free_arg_should_be_a_filter",
              ~"test::ignored_tests_result_in_ignored",
              ~"test::parse_ignored_flag",
              ~"test::sort_tests"];

        let pairs = vec::zip(expected, move filtered);

        for vec::each(pairs) |p| {
            match *p {
                (ref a, ref b) => { assert (*a == b.desc.name); }
            }
        }
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
