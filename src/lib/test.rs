// Support code for rustc's built in test runner generator. Currently,
// none of this is meant for users. It is intended to support the
// simplest interface possible for representing and running tests
// while providing a base that other test frameworks may build off of.

export test_name;
export test_fn;
export test_desc;
export test_main;

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
                     test_fn fn);

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs (generated at compile time).
fn test_main(&vec[str] args, &test_desc[] tests) {
    if (!run_tests(parse_opts(args), tests)) {
        fail "Some tests failed";
    }
}

type test_opts = rec(option::t[str] filter);

fn parse_opts(&vec[str] args) -> test_opts {
    rec(filter = if (vec::len(args) > 1u) {
            option::some(args.(1))
        } else {
            option::none
        })
}

// A simple console test runner
fn run_tests(&test_opts opts, &test_desc[] tests) -> bool {

    auto filtered_tests = filter_tests(opts, tests);

    auto out = io::stdout();

    auto total = ivec::len(filtered_tests);
    out.write_line(#fmt("running %u tests", total));

    auto passed = 0u;
    auto failed = 0u;

    for (test_desc test in filtered_tests) {
        out.write_str(#fmt("running %s ... ", test.name));
        if (run_test(test)) {
            passed += 1u;
            write_ok(out);
            out.write_line("");
        } else {
            failed += 1u;
            write_failed(out);
            out.write_line("");
        }
    }

    assert passed + failed == total;

    out.write_str(#fmt("\nresult: "));
    if (failed == 0u) {
        write_ok(out);
    } else {
        write_failed(out);
    }
    out.write_str(#fmt(". %u passed; %u failed\n\n",
                       passed, failed));

    ret true;

    fn run_test(&test_desc test) -> bool {
        test.fn();
        ret true;
    }

    fn write_ok(&io::writer out) {
        if (term::color_supported()) {
            term::fg(out.get_buf_writer(), term::color_green);
        }
        out.write_str("ok");
        if (term::color_supported()) {
            term::reset(out.get_buf_writer());
        }
     }

    fn write_failed(&io::writer out) {
        if (term::color_supported()) {
            term::fg(out.get_buf_writer(), term::color_red);
        }
        out.write_str("FAILED");
        if (term::color_supported()) {
            term::reset(out.get_buf_writer());
        }
    }
}

fn filter_tests(&test_opts opts, &test_desc[] tests) -> test_desc[] {
    if (option::is_none(opts.filter)) {
        ret tests;
    }

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

    ret ivec::filter_map(filter, tests);
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
