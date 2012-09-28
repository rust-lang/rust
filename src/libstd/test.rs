#[doc(hidden)];

// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

#[warn(deprecated_mode)];

use core::cmp::Eq;
use either::Either;
use result::{Ok, Err};
use io::WriterUtil;
use libc::size_t;
use task::TaskBuilder;
use comm = core::comm;

export TestName;
export TestFn;
export TestDesc;
export test_main;
export TestResult;
export TestOpts;
export TrOk;
export TrFailed;
export TrIgnored;
export run_tests_console;

#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];
    fn rust_sched_threads() -> libc::size_t;
}

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.
type TestName = ~str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
type TestFn = fn~();

// The definition of a single test. A test runner will run a list of
// these.
type TestDesc = {
    name: TestName,
    testfn: TestFn,
    ignore: bool,
    should_fail: bool
};

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
fn test_main(args: &[~str], tests: &[TestDesc]) {
    let opts =
        match parse_opts(args) {
          either::Left(move o) => o,
          either::Right(move m) => fail m
        };
    if !run_tests_console(&opts, tests) { fail ~"Some tests failed"; }
}

type TestOpts = {filter: Option<~str>, run_ignored: bool,
                  logfile: Option<~str>};

type OptRes = Either<TestOpts, ~str>;

// Parses command line arguments into test options
fn parse_opts(args: &[~str]) -> OptRes {
    let args_ = vec::tail(args);
    let opts = ~[getopts::optflag(~"ignored"), getopts::optopt(~"logfile")];
    let matches =
        match getopts::getopts(args_, opts) {
          Ok(move m) => m,
          Err(move f) => return either::Right(getopts::fail_str(f))
        };

    let filter =
        if vec::len(matches.free) > 0u {
            option::Some(matches.free[0])
        } else { option::None };

    let run_ignored = getopts::opt_present(matches, ~"ignored");
    let logfile = getopts::opt_maybe_str(matches, ~"logfile");

    let test_opts = {filter: filter, run_ignored: run_ignored,
                     logfile: logfile};

    return either::Left(test_opts);
}

enum TestResult { TrOk, TrFailed, TrIgnored, }

impl TestResult : Eq {
    pure fn eq(other: &TestResult) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &TestResult) -> bool { !self.eq(other) }
}

type ConsoleTestState =
    @{out: io::Writer,
      log_out: Option<io::Writer>,
      use_color: bool,
      mut total: uint,
      mut passed: uint,
      mut failed: uint,
      mut ignored: uint,
      mut failures: ~[TestDesc]};

// A simple console test runner
fn run_tests_console(opts: &TestOpts,
                     tests: &[TestDesc]) -> bool {

    fn callback(event: &TestEvent, st: ConsoleTestState) {
        debug!("callback(event=%?)", event);
        match *event {
          TeFiltered(ref filtered_tests) => {
            st.total = filtered_tests.len();
            let noun = if st.total != 1u { ~"tests" } else { ~"test" };
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
                st.passed += 1u;
                write_ok(st.out, st.use_color);
                st.out.write_line(~"");
              }
              TrFailed => {
                st.failed += 1u;
                write_failed(st.out, st.use_color);
                st.out.write_line(~"");
                st.failures.push(test);
              }
              TrIgnored => {
                st.ignored += 1u;
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
              fail(fmt!("can't open output file: %s", *s))
          }
        },
        None => None
    };

    let st =
        @{out: io::stdout(),
          log_out: log_out,
          use_color: use_color(),
          mut total: 0u,
          mut passed: 0u,
          mut failed: 0u,
          mut ignored: 0u,
          mut failures: ~[]};

    run_tests(opts, tests, |x| callback(&x, st));

    assert (st.passed + st.failed + st.ignored == st.total);
    let success = st.failed == 0u;

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

fn print_failures(st: ConsoleTestState) {
    st.out.write_line(~"\nfailures:");
    let failures = copy st.failures;
    let failures = vec::map(failures, |test| test.name);
    let failures = sort::merge_sort(|x, y| str::le(*x, *y), failures);
    for vec::each(failures) |name| {
        st.out.write_line(fmt!("    %s", *name));
    }
}

#[test]
fn should_sort_failures_before_printing_them() {
    let s = do io::with_str_writer |wr| {
        let test_a = {
            name: ~"a",
            testfn: fn~() { },
            ignore: false,
            should_fail: false
        };

        let test_b = {
            name: ~"b",
            testfn: fn~() { },
            ignore: false,
            should_fail: false
        };

        let st =
            @{out: wr,
              log_out: option::None,
              use_color: false,
              mut total: 0u,
              mut passed: 0u,
              mut failed: 0u,
              mut ignored: 0u,
              mut failures: ~[test_b, test_a]};

        print_failures(st);
    };

    let apos = str::find_str(s, ~"a").get();
    let bpos = str::find_str(s, ~"b").get();
    assert apos < bpos;
}

fn use_color() -> bool { return get_concurrency() == 1u; }

enum TestEvent {
    TeFiltered(~[TestDesc]),
    TeWait(TestDesc),
    TeResult(TestDesc, TestResult),
}

type MonitorMsg = (TestDesc, TestResult);

fn run_tests(opts: &TestOpts, tests: &[TestDesc],
             callback: fn@(+e: TestEvent)) {

    let mut filtered_tests = filter_tests(opts, tests);
    callback(TeFiltered(copy filtered_tests));

    // It's tempting to just spawn all the tests at once, but since we have
    // many tests that run in other processes we would be making a big mess.
    let concurrency = get_concurrency();
    debug!("using %u test tasks", concurrency);

    let total = vec::len(filtered_tests);
    let mut run_idx = 0;
    let mut wait_idx = 0;
    let mut done_idx = 0;

    let p = core::comm::Port();
    let ch = core::comm::Chan(p);

    while done_idx < total {
        while wait_idx < concurrency && run_idx < total {
            let test = copy filtered_tests[run_idx];
            if concurrency == 1 {
                // We are doing one test at a time so we can print the name
                // of the test before we run it. Useful for debugging tests
                // that hang forever.
                callback(TeWait(copy test));
            }
            run_test(move test, ch);
            wait_idx += 1;
            run_idx += 1;
        }

        let (test, result) = core::comm::recv(p);
        if concurrency != 1 {
            callback(TeWait(copy test));
        }
        callback(TeResult(move test, result));
        wait_idx -= 1;
        done_idx += 1;
    }
}

// Windows tends to dislike being overloaded with threads.
#[cfg(windows)]
const sched_overcommit : uint = 1u;

#[cfg(unix)]
const sched_overcommit : uint = 4u;

fn get_concurrency() -> uint {
    let threads = rustrt::rust_sched_threads() as uint;
    if threads == 1u { 1u }
    else { threads * sched_overcommit }
}

#[allow(non_implicitly_copyable_typarams)]
fn filter_tests(opts: &TestOpts,
                tests: &[TestDesc]) -> ~[TestDesc] {
    let mut filtered = vec::slice(tests, 0, tests.len());

    // Remove tests that don't match the test filter
    filtered = if opts.filter.is_none() {
        move filtered
    } else {
        let filter_str =
            match opts.filter {
          option::Some(copy f) => f,
          option::None => ~""
        };

        fn filter_fn(test: &TestDesc, filter_str: &str) ->
            Option<TestDesc> {
            if str::contains(test.name, filter_str) {
                return option::Some(copy *test);
            } else { return option::None; }
        }

        vec::filter_map(filtered, |x| filter_fn(x, filter_str))
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        move filtered
    } else {
        fn filter(test: &TestDesc) -> Option<TestDesc> {
            if test.ignore {
                return option::Some({name: test.name,
                                  testfn: copy test.testfn,
                                  ignore: false,
                                  should_fail: test.should_fail});
            } else { return option::None; }
        };

        vec::filter_map(filtered, |x| filter(x))
    };

    // Sort the tests alphabetically
    filtered = {
        pure fn lteq(t1: &TestDesc, t2: &TestDesc) -> bool {
            str::le(t1.name, t2.name)
        }
        sort::merge_sort(lteq, filtered)
    };

    move filtered
}

type TestFuture = {test: TestDesc, wait: fn@() -> TestResult};

fn run_test(+test: TestDesc, monitor_ch: comm::Chan<MonitorMsg>) {
    if test.ignore {
        core::comm::send(monitor_ch, (copy test, TrIgnored));
        return;
    }

    do task::spawn |move test| {
        let testfn = copy test.testfn;
        let mut result_future = None; // task::future_result(builder);
        task::task().unlinked().future_result(|+r| {
            result_future = Some(move r);
        }).spawn(move testfn);
        let task_result = future::get(&option::unwrap(move result_future));
        let test_result = calc_result(&test, task_result == task::Success);
        comm::send(monitor_ch, (copy test, test_result));
    };
}

fn calc_result(test: &TestDesc, task_succeeded: bool) -> TestResult {
    if task_succeeded {
        if test.should_fail { TrFailed }
        else { TrOk }
    } else {
        if test.should_fail { TrOk }
        else { TrFailed }
    }
}

#[cfg(test)]
mod tests {
    #[legacy_exports];

    #[test]
    fn do_not_run_ignored_tests() {
        fn f() { fail; }
        let desc = {
            name: ~"whatever",
            testfn: f,
            ignore: true,
            should_fail: false
        };
        let p = core::comm::Port();
        let ch = core::comm::Chan(p);
        run_test(desc, ch);
        let (_, res) = core::comm::recv(p);
        assert res != TrOk;
    }

    #[test]
    fn ignored_tests_result_in_ignored() {
        fn f() { }
        let desc = {
            name: ~"whatever",
            testfn: f,
            ignore: true,
            should_fail: false
        };
        let p = core::comm::Port();
        let ch = core::comm::Chan(p);
        run_test(desc, ch);
        let (_, res) = core::comm::recv(p);
        assert res == TrIgnored;
    }

    #[test]
    #[ignore(cfg(windows))]
    fn test_should_fail() {
        fn f() { fail; }
        let desc = {
            name: ~"whatever",
            testfn: f,
            ignore: false,
            should_fail: true
        };
        let p = core::comm::Port();
        let ch = core::comm::Chan(p);
        run_test(desc, ch);
        let (_, res) = core::comm::recv(p);
        assert res == TrOk;
    }

    #[test]
    fn test_should_fail_but_succeeds() {
        fn f() { }
        let desc = {
            name: ~"whatever",
            testfn: f,
            ignore: false,
            should_fail: true
        };
        let p = core::comm::Port();
        let ch = core::comm::Chan(p);
        run_test(desc, ch);
        let (_, res) = core::comm::recv(p);
        assert res == TrFailed;
    }

    #[test]
    fn first_free_arg_should_be_a_filter() {
        let args = ~[~"progname", ~"filter"];
        let opts = match parse_opts(args) {
          either::Left(copy o) => o,
          _ => fail ~"Malformed arg in first_free_arg_should_be_a_filter"
        };
        assert ~"filter" == opts.filter.get();
    }

    #[test]
    fn parse_ignored_flag() {
        let args = ~[~"progname", ~"filter", ~"--ignored"];
        let opts = match parse_opts(args) {
          either::Left(copy o) => o,
          _ => fail ~"Malformed arg in parse_ignored_flag"
        };
        assert (opts.run_ignored);
    }

    #[test]
    fn filter_for_ignored_option() {
        // When we run ignored tests the test filter should filter out all the
        // unignored tests and flip the ignore flag on the rest to false

        let opts = {filter: option::None, run_ignored: true,
            logfile: option::None};
        let tests =
            ~[{name: ~"1", testfn: fn~() { },
               ignore: true, should_fail: false},
             {name: ~"2", testfn: fn~() { },
              ignore: false, should_fail: false}];
        let filtered = filter_tests(&opts, tests);

        assert (vec::len(filtered) == 1u);
        assert (filtered[0].name == ~"1");
        assert (filtered[0].ignore == false);
    }

    #[test]
    fn sort_tests() {
        let opts = {filter: option::None, run_ignored: false,
            logfile: option::None};

        let names =
            ~[~"sha1::test", ~"int::test_to_str", ~"int::test_pow",
             ~"test::do_not_run_ignored_tests",
             ~"test::ignored_tests_result_in_ignored",
             ~"test::first_free_arg_should_be_a_filter",
             ~"test::parse_ignored_flag", ~"test::filter_for_ignored_option",
             ~"test::sort_tests"];
        let tests =
        {
            let testfn = fn~() { };
            let mut tests = ~[];
            for vec::each(names) |name| {
                let test = {name: *name, testfn: copy testfn, ignore: false,
                            should_fail: false};
                tests.push(test);
            }
            tests
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

        let pairs = vec::zip(expected, filtered);

        for vec::each(pairs) |p| {
            match *p {
                (ref a, ref b) => { assert (*a == b.name); }
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
