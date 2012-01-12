import core::*;

import std::test;
import str;
import option;
import either;
import vec;

#[test]
fn do_not_run_ignored_tests() {
    fn f() { fail; }
    let desc = {name: "whatever", fn: f, ignore: true, should_fail: false};
    let future = test::run_test(desc, test::default_test_to_task);
    let result = future.wait();
    assert result != test::tr_ok;
}

#[test]
fn ignored_tests_result_in_ignored() {
    fn f() { }
    let desc = {name: "whatever", fn: f, ignore: true, should_fail: false};
    let res = test::run_test(desc, test::default_test_to_task).wait();
    assert (res == test::tr_ignored);
}

#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_should_fail() {
    fn f() { fail; }
    let desc = {name: "whatever", fn: f, ignore: false, should_fail: true};
    let res = test::run_test(desc, test::default_test_to_task).wait();
    assert res == test::tr_ok;
}

#[test]
fn test_should_fail_but_succeeds() {
    fn f() { }
    let desc = {name: "whatever", fn: f, ignore: false, should_fail: true};
    let res = test::run_test(desc, test::default_test_to_task).wait();
    assert res == test::tr_failed;
}

#[test]
fn first_free_arg_should_be_a_filter() {
    let args = ["progname", "filter"];
    check (vec::is_not_empty(args));
    let opts = alt test::parse_opts(args) { either::left(o) { o } };
    assert (str::eq("filter", option::get(opts.filter)));
}

#[test]
fn parse_ignored_flag() {
    let args = ["progname", "filter", "--ignored"];
    check (vec::is_not_empty(args));
    let opts = alt test::parse_opts(args) { either::left(o) { o } };
    assert (opts.run_ignored);
}

#[test]
fn filter_for_ignored_option() {
    // When we run ignored tests the test filter should filter out all the
    // unignored tests and flip the ignore flag on the rest to false

    let opts = {filter: option::none, run_ignored: true};
    let tests =
        [{name: "1", fn: fn@() { }, ignore: true, should_fail: false},
         {name: "2", fn: fn@() { }, ignore: false, should_fail: false}];
    let filtered = test::filter_tests(opts, tests);

    assert (vec::len(filtered) == 1u);
    assert (filtered[0].name == "1");
    assert (filtered[0].ignore == false);
}

#[test]
fn sort_tests() {
    let opts = {filter: option::none, run_ignored: false};

    let names =
        ["sha1::test", "int::test_to_str", "int::test_pow",
         "test::do_not_run_ignored_tests",
         "test::ignored_tests_result_in_ignored",
         "test::first_free_arg_should_be_a_filter",
         "test::parse_ignored_flag", "test::filter_for_ignored_option",
         "test::sort_tests"];
    let tests =
        {
            let testfn = fn@() { };
            let tests = [];
            for name: str in names {
                let test = {name: name, fn: testfn, ignore: false,
                  should_fail: false};
                tests += [test];
            }
            tests
        };
    let filtered = test::filter_tests(opts, tests);

    let expected =
        ["int::test_pow", "int::test_to_str", "sha1::test",
         "test::do_not_run_ignored_tests", "test::filter_for_ignored_option",
         "test::first_free_arg_should_be_a_filter",
         "test::ignored_tests_result_in_ignored", "test::parse_ignored_flag",
         "test::sort_tests"];

    check (vec::same_length(expected, filtered));
    let pairs = vec::zip(expected, filtered);


    for (a, b) in pairs { assert (a == b.name); }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
