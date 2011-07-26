


// xfail-stage0

// Regression test for issue #377
fn main() {
    auto a = { auto b = rec(a=3); b };
    assert (a.a == 3);
    auto c = { auto d = rec(v=3); d };
    assert (c.v == 3);
}