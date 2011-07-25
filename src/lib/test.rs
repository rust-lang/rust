// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

import sort = sort::ivector;
import getenv = generic_os::getenv;

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
export run_test;
export filter_tests;
export parse_opts;

// The name of a test. By convention this follows the rules for rust
// paths, i.e it should be a series of identifiers seperated by double
// colons. This way if some test runner wants to arrange the tests
// heirarchically it may.
type test_name = str;

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function fails then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into tasks.
type test_fn = fn();

// The definition of a single test. A test runner will run a list of
// these.
type test_desc = rec(test_name name,
                     test_fn fn,
                     bool ignore);

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
fn test_main(&vec[str] args, &test_desc[] tests) {
    auto ivec_args = {
        auto iargs = ~[];
        for (str arg in args) {
            iargs += ~[arg]
        }
        iargs
    };
    check ivec::is_not_empty(ivec_args);
    auto opts = alt (parse_opts(ivec_args)) {
        either::left(?o) { o }
        either::right(?m) { fail m }
    };
    if (!run_tests_console(opts, tests)) {
        fail "Some tests failed";
    }
}

type test_opts = rec(option::t[str] filter,
                     bool run_ignored);

type opt_res = either::t[test_opts, str];

// Parses command line arguments into test options
fn parse_opts(&str[] args) : ivec::is_not_empty(args) -> opt_res {

    // FIXME (#649): Shouldn't have to check here
    check ivec::is_not_empty(args);
    auto args_ = ivec::tail(args);
    auto opts = ~[getopts::optflag("ignored")];
    auto match = alt (getopts::getopts_ivec(args_, opts)) {
        getopts::success(?m) { m }
        getopts::failure(?f) { ret either::right(getopts::fail_str(f)) }
    };

    auto filter = if (vec::len(match.free) > 0u) {
        option::some(match.free.(0))
    } else {
        option::none
    };

    auto run_ignored = getopts::opt_present(match, "ignored");

    auto test_opts = rec(filter = filter,
                         run_ignored = run_ignored);

    ret either::left(test_opts);
}

tag test_result {
    tr_ok;
    tr_failed;
    tr_ignored;
}

// A simple console test runner
fn run_tests_console(&test_opts opts, &test_desc[] tests) -> bool {

    auto filtered_tests = filter_tests(opts, tests);

    auto out = io::stdout();

    auto total = ivec::len(filtered_tests);
    out.write_line(#fmt("running %u tests", total));

    auto futures = ~[];

    auto passed = 0u;
    auto failed = 0u;
    auto ignored = 0u;

    auto failures = ~[];

    // It's tempting to just spawn all the tests at once but that doesn't
    // provide a great user experience because you might sit waiting for the
    // result of a particular test for an unusually long amount of time.
    auto concurrency = get_concurrency();
    log #fmt("using %u test tasks", concurrency);
    auto run_idx = 0u;
    auto wait_idx = 0u;

    while (wait_idx < total) {
        while (ivec::len(futures) < concurrency
               && run_idx < total) {
            futures += ~[run_test(filtered_tests.(run_idx))];
            run_idx += 1u;
        }

        auto future = futures.(0);
        out.write_str(#fmt("running %s ... ", future.test.name));
        auto result = future.wait();
        alt (result) {
            tr_ok {
                passed += 1u;
                write_ok(out, concurrency);
                out.write_line("");
            }
            tr_failed {
                failed += 1u;
                write_failed(out, concurrency);
                out.write_line("");
                failures += ~[future.test];
            }
            tr_ignored {
                ignored += 1u;
                write_ignored(out, concurrency);
                out.write_line("");
            }
        }
        futures = ivec::slice(futures, 1u, ivec::len(futures));
        wait_idx += 1u;
    }

    assert passed + failed + ignored == total;
    auto success = failed == 0u;

    if (!success) {
        out.write_line("\nfailures:");
        for (test_desc test in failures) {
            out.write_line(#fmt("    %s", test.name));
        }
    }

    out.write_str(#fmt("\nresult: "));
    if (success) {
        write_ok(out, concurrency);
    } else {
        write_failed(out, concurrency);
    }
    out.write_str(#fmt(". %u passed; %u failed; %u ignored\n\n",
                       passed, failed, ignored));

    ret success;

    fn write_ok(&io::writer out, uint concurrency) {
        write_pretty(out, "ok", term::color_green, concurrency);
     }

    fn write_failed(&io::writer out, uint concurrency) {
        write_pretty(out, "FAILED", term::color_red, concurrency);
    }

    fn write_ignored(&io::writer out, uint concurrency) {
        write_pretty(out, "ignored", term::color_yellow, concurrency);
    }

    fn write_pretty(&io::writer out, &str word, u8 color,
                   uint concurrency) {
        // In the presence of concurrency, outputing control characters
        // can cause some crazy artifacting
        if (concurrency == 1u && term::color_supported()) {
            term::fg(out.get_buf_writer(), color);
        }
        out.write_str(word);
        if (concurrency == 1u && term::color_supported()) {
            term::reset(out.get_buf_writer());
        }
    }
}

fn get_concurrency() -> uint {
    alt getenv("RUST_THREADS") {
      option::some(?t) {
        auto threads = uint::parse_buf(str::bytes(t), 10u);
        threads > 0u ? threads : 1u
      }
      option::none {
        1u
      }
    }
}

fn filter_tests(&test_opts opts, &test_desc[] tests) -> test_desc[] {
    auto filtered = tests;

    // Remove tests that don't match the test filter
    filtered = if (option::is_none(opts.filter)) {
        filtered
    } else {
        auto filter_str = alt opts.filter { option::some(?f) { f }
                                            option::none { "" } };

        auto filter = bind fn(&test_desc test,
                              str filter_str) -> option::t[test_desc] {
            if (str::find(test.name, filter_str) >= 0) {
                ret option::some(test);
            } else {
                ret option::none;
            }
        } (_, filter_str);

        ivec::filter_map(filter, filtered)
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if (!opts.run_ignored) {
        filtered
    } else {
        auto filter = fn(&test_desc test) -> option::t[test_desc] {
            if (test.ignore) {
                ret option::some(rec(name = test.name,
                                     fn = test.fn,
                                     ignore = false));
            } else {
                ret option::none;
            }
        };

        ivec::filter_map(filter, filtered)
    };

    // Sort the tests alphabetically
    filtered = {
        fn lteq(&test_desc t1, &test_desc t2) -> bool {
            str::lteq(t1.name, t2.name)
        }
        sort::merge_sort(lteq, filtered)
    };

    ret filtered;
}

type test_future = rec(test_desc test,
                       @fn() fnref,
                       fn() -> test_result wait);

fn run_test(&test_desc test) -> test_future {
    // FIXME: Because of the unsafe way we're passing the test function
    // to the test task, we need to make sure we keep a reference to that
    // function around for longer than the lifetime of the task. To that end
    // we keep the function boxed in the test future.
    auto fnref = @test.fn;
    if (!test.ignore) {
        auto test_task = run_test_fn_in_task(*fnref);
        ret rec(test = test,
                fnref = fnref,
                wait = bind fn(&task test_task) -> test_result {
                    alt (task::join(test_task)) {
                      task::tr_success { tr_ok }
                      task::tr_failure { tr_failed }
                    }
                } (test_task));
    } else {
        ret rec(test = test,
                fnref = fnref,
                wait = fn() -> test_result { tr_ignored });
    }
}

native "rust" mod rustrt {
    fn hack_allow_leaks();
}

// We need to run our tests in another task in order to trap test failures.
// But, at least currently, functions can't be used as spawn arguments so
// we've got to treat our test functions as unsafe pointers.
fn run_test_fn_in_task(&fn() f) -> task {
    fn run_task(*mutable fn() fptr) {
        // If this task fails we don't want that failure to propagate to the
        // test runner or else we couldn't keep running tests
        task::unsupervise();

        // FIXME (236): Hack supreme - unwinding doesn't work yet so if this
        // task fails memory will not be freed correctly. This turns off the
        // sanity checks in the runtime's memory region for the task, so that
        // the test runner can continue.
        rustrt::hack_allow_leaks();

        // Run the test
        (*fptr)()
    }
    auto fptr = ptr::addr_of(f);
    ret spawn run_task(fptr);
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
