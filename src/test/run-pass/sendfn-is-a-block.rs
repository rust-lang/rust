#[legacy_modes];

fn test(f: fn(uint) -> uint) -> uint {
    return f(22u);
}

fn main() {
    let y = test(fn~(x: uint) -> uint { return 4u * x; });
    assert y == 88u;
}
