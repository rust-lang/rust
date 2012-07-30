fn foo(x: &uint) -> uint {
    *x
}

fn main() {
    let p = @22u;
    let r = foo(p);
    debug!{"r=%u", r};
    assert r == 22u;
}
