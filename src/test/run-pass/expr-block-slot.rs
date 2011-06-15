


// xfail-stage0

// Regression test for issue #377
fn main() {
    auto a = { auto b = tup(3); b };
    assert (a._0 == 3);
    auto c = { auto d = rec(v=3); d };
    assert (c.v == 3);
}