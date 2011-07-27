import std::test;
import std::str;
import std::option;
import std::either;
import std::ivec;

#[test]
fn do_not_run_ignored_tests() {
    auto ran = @mutable false;
    auto f = bind fn(@mutable bool ran) {
        *ran = true;
    } (ran);

    auto desc = rec(name = "whatever",
                    fn = f,
                    ignore = true);

    test::run_test(desc, test::default_test_to_task);

    assert ran == false;
}

#[test]
fn ignored_tests_result_in_ignored() {
    fn f() { }
    auto desc = rec(name = "whatever",
                    fn = f,
                    ignore = true);
    auto res = test::run_test(desc, test::default_test_to_task).wait();
    assert res == test::tr_ignored;
}

#[test]
fn first_free_arg_should_be_a_filter() {
    auto args = ~["progname", "filter"];
    check ivec::is_not_empty(args);
    auto opts = alt test::parse_opts(args) { either::left(?o) { o } };
    assert str::eq("filter", option::get(opts.filter));
}

#[test]
fn parse_ignored_flag() {
    auto args = ~["progname", "filter", "--ignored"];
    check ivec::is_not_empty(args);
    auto opts = alt test::parse_opts(args) { either::left(?o) { o } };
    assert opts.run_ignored;
}

#[test]
fn filter_for_ignored_option() {
    // When we run ignored tests the test filter should filter out all the
    // unignored tests and flip the ignore flag on the rest to false

    auto opts = rec(filter = option::none,
                    run_ignored = true);
    auto tests = ~[rec(name = "1",
                       fn = fn() {},
                       ignore = true),
                   rec(name = "2",
                       fn = fn() {},
                       ignore = false)];
    auto filtered = test::filter_tests(opts, tests);

    assert ivec::len(filtered) == 1u;
    assert filtered.(0).name == "1";
    assert filtered.(0).ignore == false;
}

#[test]
fn sort_tests() {
    auto opts = rec(filter = option::none,
                    run_ignored = false);

    auto names = ~["sha1::test",
                   "int::test_to_str",
                   "int::test_pow",
                   "test::do_not_run_ignored_tests",
                   "test::ignored_tests_result_in_ignored",
                   "test::first_free_arg_should_be_a_filter",
                   "test::parse_ignored_flag",
                   "test::filter_for_ignored_option",
                   "test::sort_tests"];
    auto tests = {
        auto testfn = fn() {};
        auto tests = ~[];
        for (str name in names) {
            auto test = rec(name = name,
                            fn = testfn,
                            ignore = false);
            tests += ~[test];
        }
        tests
    };
    auto filtered = test::filter_tests(opts, tests);

    auto expected = ~["int::test_pow",
                      "int::test_to_str",
                      "sha1::test",
                      "test::do_not_run_ignored_tests",
                      "test::filter_for_ignored_option",
                      "test::first_free_arg_should_be_a_filter",
                      "test::ignored_tests_result_in_ignored",
                      "test::parse_ignored_flag",
                      "test::sort_tests"];

    auto pairs = ivec::zip(expected, filtered);

    for (rec(str _0, test::test_desc _1) p in pairs) {
        assert p._0 == p._1.name;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
