import std::test;

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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
