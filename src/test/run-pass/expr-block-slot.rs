// Regression test for issue #377
fn main() {
    let a = { let b = {a: 3}; b };
    assert (a.a == 3);
    let c = { let d = {v: 3}; d };
    assert (c.v == 3);
}
