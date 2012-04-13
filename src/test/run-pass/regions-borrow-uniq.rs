fn foo(x: &uint) -> uint {
    *x
}

fn main() {
    let p = ~3u;
    let r = foo(p);
    assert r == 3u;
}
