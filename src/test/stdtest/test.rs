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

    auto res = test::run_test(desc);

    assert ran == false;
}

#[test]
fn ignored_tests_result_in_ignored() {
    fn f() { }
    auto desc = rec(name = "whatever",
                    fn = f,
                    ignore = true);
    auto res = test::run_test(desc);
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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
