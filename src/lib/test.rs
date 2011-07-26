// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

import sort = sort::ivector;
import generic_os::getenv;

export test_name;
export test_fn;
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

native "rust" mod rustrt {
    fn hack_allow_leaks();
    fn sched_threads() -> uint;
}


// The name of a test. By convention this follows the rules for rust
// paths, i.e it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// heirarchically it may.
type test_name = str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
type test_fn = fn() ;

// The definition of a single test. A test runner will run a list of
// these.
type test_desc = {name: test_name, fn: test_fn, ignore: bool};

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
fn test_main(args: &vec[str], tests: &test_desc[]) {
    let ivec_args =
        { let iargs = ~[]; for arg: str  in args { iargs += ~[arg] } iargs };
    check (ivec::is_not_empty(ivec_args));
    let opts =
        alt parse_opts(ivec_args) {
          either::left(o) { o }
          either::right(m) { fail m }
        };
    if !run_tests_console(opts, tests) { fail "Some tests failed"; }
}

type test_opts = {filter: option::t[str], run_ignored: bool};

type opt_res = either::t[test_opts, str];

// Parses command line arguments into test options
fn parse_opts(args: &str[]) : ivec::is_not_empty(args) -> opt_res {

    // FIXME (#649): Shouldn't have to check here
    check (ivec::is_not_empty(args));
    let args_ = ivec::tail(args);
    let opts = ~[getopts::optflag("ignored")];
    let match =
        alt getopts::getopts_ivec(args_, opts) {
          getopts::success(m) { m }
          getopts::failure(f) { ret either::right(getopts::fail_str(f)) }
        };

    let filter =
        if vec::len(match.free) > 0u {
            option::some(match.free.(0))
        } else { option::none };

    let run_ignored = getopts::opt_present(match, "ignored");

    let test_opts = {filter: filter, run_ignored: run_ignored};

    ret either::left(test_opts);
}

tag test_result { tr_ok; tr_failed; tr_ignored; }

// To get isolation and concurrency tests have to be run in their own tasks.
// In cases where test functions and closures it is not ok to just dump them
// into a task and run them, so this transformation gives the caller a chance
// to create the test task.
type test_to_task = fn(&fn()) -> task ;

// A simple console test runner
fn run_tests_console(opts: &test_opts, tests: &test_desc[]) -> bool {
    run_tests_console_(opts, tests, default_test_to_task)
}

fn run_tests_console_(opts: &test_opts, tests: &test_desc[],
                      to_task: &test_to_task) -> bool {

    type test_state = @{
        out: io::writer,
        use_color: bool,
        mutable total: uint,
        mutable passed: uint,
        mutable failed: uint,
        mutable ignored: uint,
        mutable failures: test_desc[]
    };

    fn callback(event: testevent, st: test_state) {
        alt event {
          te_filtered(filtered_tests) {
            st.total = ivec::len(filtered_tests);
            st.out.write_line(#fmt("\nrunning %u tests", st.total));
          }
          te_result(test, result) {
            st.out.write_str(#fmt("test %s ... ", test.name));
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
                st.failures += ~[test];
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

    let st = @{
        out: io::stdout(),
        use_color: use_color(),
        mutable total: 0u,
        mutable passed: 0u,
        mutable failed: 0u,
        mutable ignored: 0u,
        mutable failures: ~[]
    };

    run_tests(opts, tests, to_task,
              bind callback(_, st));

    assert st.passed + st.failed + st.ignored == st.total;
    let success = st.failed == 0u;

    if !success {
        st.out.write_line("\nfailures:");
        for test: test_desc in st.failures {
            let testname = test.name; // Satisfy alias analysis
            st.out.write_line(#fmt("    %s", testname));
        }
    }

    st.out.write_str(#fmt("\nresult: "));
    if success {
        write_ok(st.out, st.use_color);
    } else { write_failed(st.out, st.use_color); }
    st.out.write_str(#fmt(". %u passed; %u failed; %u ignored\n\n",
                       st.passed, st.failed, st.ignored));

    ret success;

    fn write_ok(out: &io::writer, use_color: bool) {
        write_pretty(out, "ok", term::color_green, use_color);
    }

    fn write_failed(out: &io::writer, use_color: bool) {
        write_pretty(out, "FAILED", term::color_red, use_color);
    }

    fn write_ignored(out: &io::writer, use_color: bool) {
        write_pretty(out, "ignored", term::color_yellow, use_color);
    }

    fn write_pretty(out: &io::writer, word: &str, color: u8,
                    use_color: bool) {
        if use_color && term::color_supported() {
            term::fg(out.get_buf_writer(), color);
        }
        out.write_str(word);
        if use_color && term::color_supported() {
            term::reset(out.get_buf_writer());
        }
    }
}

fn use_color() -> bool {
    ret get_concurrency() == 1u;
}

tag testevent {
    te_filtered(test_desc[]);
    te_result(test_desc, test_result);
}

fn run_tests(opts: &test_opts, tests: &test_desc[],
             to_task: &test_to_task, callback: fn(testevent)) {

    let filtered_tests = filter_tests(opts, tests);

    callback(te_filtered(filtered_tests));

    // It's tempting to just spawn all the tests at once but that doesn't
    // provide a great user experience because you might sit waiting for the
    // result of a particular test for an unusually long amount of time.
    let concurrency = get_concurrency();
    log #fmt("using %u test tasks", concurrency);
    let total = ivec::len(filtered_tests);
    let run_idx = 0u;
    let wait_idx = 0u;
    let futures = ~[];

    while wait_idx < total {
        while ivec::len(futures) < concurrency && run_idx < total {
            futures += ~[run_test(filtered_tests.(run_idx), to_task)];
            run_idx += 1u;
        }

        let future = futures.(0);
        let result = future.wait();
        callback(te_result(future.test, result));
        futures = ivec::slice(futures, 1u, ivec::len(futures));
        wait_idx += 1u;
    }
}

fn get_concurrency() -> uint { rustrt::sched_threads() }

fn filter_tests(opts: &test_opts, tests: &test_desc[]) -> test_desc[] {
    let filtered = tests;

    // Remove tests that don't match the test filter
    filtered =
        if option::is_none(opts.filter) {
            filtered
        } else {
            let filter_str =
                alt opts.filter {
                  option::some(f) { f }
                  option::none. { "" }
                };

            let filter =
                bind fn (test: &test_desc, filter_str: str) ->
                        option::t[test_desc] {
                         if str::find(test.name, filter_str) >= 0 {
                             ret option::some(test);
                         } else { ret option::none; }
                     }(_, filter_str);


            ivec::filter_map(filter, filtered)
        };

    // Maybe pull out the ignored test and unignore them
    filtered =
        if !opts.run_ignored {
            filtered
        } else {
            let filter =
                fn (test: &test_desc) -> option::t[test_desc] {
                    if test.ignore {
                        ret option::some({name: test.name,
                                          fn: test.fn,
                                          ignore: false});
                    } else { ret option::none; }
                };


            ivec::filter_map(filter, filtered)
        };

    // Sort the tests alphabetically
    filtered =
        {
            fn lteq(t1: &test_desc, t2: &test_desc) -> bool {
                str::lteq(t1.name, t2.name)
            }
            sort::merge_sort(lteq, filtered)
        };

    ret filtered;
}

type test_future =
    {test: test_desc, fnref: @fn() , wait: fn() -> test_result };

fn run_test(test: &test_desc, to_task: &test_to_task) -> test_future {
    // FIXME: Because of the unsafe way we're passing the test function
    // to the test task, we need to make sure we keep a reference to that
    // function around for longer than the lifetime of the task. To that end
    // we keep the function boxed in the test future.
    let fnref = @test.fn;
    if !test.ignore {
        let test_task = to_task(*fnref);
        ret {test: test,
             fnref: fnref,
             wait:
                 bind fn (test_task: &task) -> test_result {
                          alt task::join(test_task) {
                            task::tr_success. { tr_ok }
                            task::tr_failure. { tr_failed }
                          }
                      }(test_task)};
    } else {
        ret {test: test,
             fnref: fnref,
             wait: fn () -> test_result { tr_ignored }};
    }
}

// We need to run our tests in another task in order to trap test failures.
// But, at least currently, functions can't be used as spawn arguments so
// we've got to treat our test functions as unsafe pointers.  This function
// only works with functions that don't contain closures.
fn default_test_to_task(f: &fn()) -> task {
    fn run_task(fptr: *mutable fn() ) {
        configure_test_task();
        // Run the test
        (*fptr)()
    }
    let fptr = ptr::addr_of(f);
    ret spawn run_task(fptr);
}

// Call from within a test task to make sure it's set up correctly
fn configure_test_task() {
    // If this task fails we don't want that failure to propagate to the
    // test runner or else we couldn't keep running tests
    task::unsupervise();

    // FIXME (236): Hack supreme - unwinding doesn't work yet so if this
    // task fails memory will not be freed correctly. This turns off the
    // sanity checks in the runtime's memory region for the task, so that
    // the test runner can continue.
    rustrt::hack_allow_leaks();
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
