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
fn test_main(&vec[str] args, &test_desc[] tests) -> int {
    if (run_tests(tests)) {
        ret 0;
    } else {
        ret -1;
    }
}

fn run_tests(&test_desc[] tests) -> bool {
    auto out = io::stdout();

    for (test_desc test in tests) {
        out.write_line("running " + test.name);
    }

    ret true;
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
