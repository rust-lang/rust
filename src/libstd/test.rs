// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

import core::comm;
import core::task;
import task::task;
import core::option;
import core::either;
import core::vec;

export test_name;
export test_fn;
export default_test_fn;
export test_desc;
export test_main;
export test_result;
export test_opts;
export tr_ok;
export tr_failed;
export tr_ignored;
export run_tests_console;
export run_tests_console_;
export run_test;
export filter_tests;
export parse_opts;
export test_to_task;
export default_test_to_task;
export configure_test_task;
export joinable;

#[abi = "cdecl"]
native mod rustrt {
    fn sched_threads() -> uint;
}


// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.
type test_name = str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
type test_fn<T> = T;

type default_test_fn = test_fn<fn()>;

// The definition of a single test. A test runner will run a list of
// these.
type test_desc<T> = {
    name: test_name,
    fn: test_fn<T>,
    ignore: bool,
    should_fail: bool
};

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
fn test_main(args: [str], tests: [test_desc<default_test_fn>]) {
    check (vec::is_not_empty(args));
    let opts =
        alt parse_opts(args) {
          either::left(o) { o }
          either::right(m) { fail m }
        };
    if !run_tests_console(opts, tests) { fail "Some tests failed"; }
}

type test_opts = {filter: option::t<str>, run_ignored: bool};

type opt_res = either::t<test_opts, str>;

// Parses command line arguments into test options
fn parse_opts(args: [str]) : vec::is_not_empty(args) -> opt_res {

    let args_ = vec::tail(args);
    let opts = [getopts::optflag("ignored")];
    let match =
        alt getopts::getopts(args_, opts) {
          getopts::success(m) { m }
          getopts::failure(f) { ret either::right(getopts::fail_str(f)) }
        };

    let filter =
        if vec::len(match.free) > 0u {
            option::some(match.free[0])
        } else { option::none };

    let run_ignored = getopts::opt_present(match, "ignored");

    let test_opts = {filter: filter, run_ignored: run_ignored};

    ret either::left(test_opts);
}

tag test_result { tr_ok; tr_failed; tr_ignored; }

type joinable = (task, comm::port<task::task_notification>);

// To get isolation and concurrency tests have to be run in their own tasks.
// In cases where test functions are closures it is not ok to just dump them
// into a task and run them, so this transformation gives the caller a chance
// to create the test task.
type test_to_task<T> = fn@(test_fn<T>) -> joinable;

// A simple console test runner
fn run_tests_console(opts: test_opts,
                         tests: [test_desc<default_test_fn>]) -> bool {
    run_tests_console_(opts, tests, default_test_to_task)
}

fn run_tests_console_<copy T>(opts: test_opts, tests: [test_desc<T>],
                              to_task: test_to_task<T>) -> bool {

    type test_state =
        @{out: io::writer,
          use_color: bool,
          mutable total: uint,
          mutable passed: uint,
          mutable failed: uint,
          mutable ignored: uint,
          mutable failures: [test_desc<T>]};

    fn callback<copy T>(event: testevent<T>, st: test_state) {
        alt event {
          te_filtered(filtered_tests) {
            st.total = vec::len(filtered_tests);
            st.out.write_line(#fmt["\nrunning %u tests", st.total]);
          }
          te_wait(test) { st.out.write_str(#fmt["test %s ... ", test.name]); }
          te_result(test, result) {
            alt result {
              tr_ok. {
                st.passed += 1u;
                write_ok(st.out, st.use_color);
                st.out.write_line("");
              }
              tr_failed. {
                st.failed += 1u;
                write_failed(st.out, st.use_color);
                st.out.write_line("");
                st.failures += [test];
              }
              tr_ignored. {
                st.ignored += 1u;
                write_ignored(st.out, st.use_color);
                st.out.write_line("");
              }
            }
          }
        }
    }

    let st =
        @{out: io::stdout(),
          use_color: use_color(),
          mutable total: 0u,
          mutable passed: 0u,
          mutable failed: 0u,
          mutable ignored: 0u,
          mutable failures: []};

    run_tests(opts, tests, to_task, bind callback(_, st));

    assert (st.passed + st.failed + st.ignored == st.total);
    let success = st.failed == 0u;

    if !success {
        st.out.write_line("\nfailures:");
        for test: test_desc<T> in st.failures {
            let testname = test.name; // Satisfy alias analysis
            st.out.write_line(#fmt["    %s", testname]);
        }
    }

    st.out.write_str(#fmt["\nresult: "]);
    if success {
        // There's no parallelism at this point so it's safe to use color
        write_ok(st.out, true);
    } else { write_failed(st.out, true); }
    st.out.write_str(#fmt[". %u passed; %u failed; %u ignored\n\n", st.passed,
                          st.failed, st.ignored]);

    ret success;

    fn write_ok(out: io::writer, use_color: bool) {
        write_pretty(out, "ok", term::color_green, use_color);
    }

    fn write_failed(out: io::writer, use_color: bool) {
        write_pretty(out, "FAILED", term::color_red, use_color);
    }

    fn write_ignored(out: io::writer, use_color: bool) {
        write_pretty(out, "ignored", term::color_yellow, use_color);
    }

    fn write_pretty(out: io::writer, word: str, color: u8, use_color: bool) {
        if use_color && term::color_supported() {
            term::fg(out.get_buf_writer(), color);
        }
        out.write_str(word);
        if use_color && term::color_supported() {
            term::reset(out.get_buf_writer());
        }
    }
}

fn use_color() -> bool { ret get_concurrency() == 1u; }

tag testevent<T> {
    te_filtered([test_desc<T>]);
    te_wait(test_desc<T>);
    te_result(test_desc<T>, test_result);
}

fn run_tests<copy T>(opts: test_opts, tests: [test_desc<T>],
                     to_task: test_to_task<T>,
                     callback: fn@(testevent<T>)) {

    let filtered_tests = filter_tests(opts, tests);
    callback(te_filtered(filtered_tests));

    // It's tempting to just spawn all the tests at once but that doesn't
    // provide a great user experience because you might sit waiting for the
    // result of a particular test for an unusually long amount of time.
    let concurrency = get_concurrency();
    log #fmt["using %u test tasks", concurrency];
    let total = vec::len(filtered_tests);
    let run_idx = 0u;
    let wait_idx = 0u;
    let futures = [];

    while wait_idx < total {
        while vec::len(futures) < concurrency && run_idx < total {
            futures += [run_test(filtered_tests[run_idx], to_task)];
            run_idx += 1u;
        }

        let future = futures[0];
        callback(te_wait(future.test));
        let result = future.wait();
        callback(te_result(future.test, result));
        futures = vec::slice(futures, 1u, vec::len(futures));
        wait_idx += 1u;
    }
}

fn get_concurrency() -> uint { rustrt::sched_threads() }

fn filter_tests<copy T>(opts: test_opts,
                        tests: [test_desc<T>]) -> [test_desc<T>] {
    let filtered = tests;

    // Remove tests that don't match the test filter
    filtered = if option::is_none(opts.filter) {
        filtered
    } else {
        let filter_str =
            alt opts.filter {
          option::some(f) { f }
          option::none. { "" }
        };

        fn filter_fn<copy T>(test: test_desc<T>, filter_str: str) ->
            option::t<test_desc<T>> {
            if str::find(test.name, filter_str) >= 0 {
                ret option::some(test);
            } else { ret option::none; }
        }

        let filter = bind filter_fn(_, filter_str);

        vec::filter_map(filter, filtered)
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        filtered
    } else {
        fn filter<copy T>(test: test_desc<T>) -> option::t<test_desc<T>> {
            if test.ignore {
                ret option::some({name: test.name,
                                  fn: test.fn,
                                  ignore: false,
                                  should_fail: test.should_fail});
            } else { ret option::none; }
        };

        vec::filter_map(bind filter(_), filtered)
    };

    // Sort the tests alphabetically
    filtered =
        {
            fn lteq<T>(t1: test_desc<T>, t2: test_desc<T>) -> bool {
                str::lteq(t1.name, t2.name)
            }
            sort::merge_sort(bind lteq(_, _), filtered)
        };

    ret filtered;
}

type test_future<T> = {test: test_desc<T>, wait: fn@() -> test_result};

fn run_test<copy T>(test: test_desc<T>,
                    to_task: test_to_task<T>) -> test_future<T> {
    if test.ignore {
        ret {test: test, wait: fn () -> test_result { tr_ignored }};
    }

    let test_task = to_task(test.fn);
    ret {test: test,
         wait:
             bind fn (test_task: joinable, should_fail: bool) -> test_result {
                  alt task::join(test_task) {
                    task::tr_success. {
                      if should_fail { tr_failed }
                      else { tr_ok }
                    }
                    task::tr_failure. {
                      if should_fail { tr_ok }
                      else { tr_failed }
                    }
                  }
              }(test_task, test.should_fail)};
}

// We need to run our tests in another task in order to trap test failures.
// This function only works with functions that don't contain closures.
fn default_test_to_task(&&f: default_test_fn) -> joinable {
    fn run_task(f: default_test_fn) {
        configure_test_task();
        f();
    }
    ret task::spawn_joinable(copy f, run_task);
}

// Call from within a test task to make sure it's set up correctly
fn configure_test_task() {
    // If this task fails we don't want that failure to propagate to the
    // test runner or else we couldn't keep running tests
    task::unsupervise();
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
