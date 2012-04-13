// xfail-test it don't work yet

fn foo(x: [uint]/&) -> uint {
    x[0]
}

fn main() {
    let p = [22u]/@;
    let r = foo(p);
    assert r == 22u;
}
